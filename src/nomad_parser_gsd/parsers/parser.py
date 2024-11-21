import os
import sys

#! TODO: Why is TYPE_CHECKING False?
from typing import TYPE_CHECKING, List, Iterable, Union

if not TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from nomad.metainfo import (
        Context,
        Section,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

from ase.symbols import symbols2numbers
from ase.utils import (
    formula_hill,
)  # TODO: use to generate chemical formula if symbols2numbers is True?
from collections import defaultdict
import h5py
from h5py import Group
from nomad_simulations.schema_packages.particles_state import ParticlesState
import nomad_simulations.schema_packages.properties.energies as energy_module
import nomad_simulations.schema_packages.properties.forces as force_module
import numpy as np

import structlog

from nomad_parser_gsd.parsers.mdparserutils import MDParser
from nomad.config import config
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.parsing.file_parser import FileParser
from nomad.units import ureg

# nomad-simulations
from nomad_simulations.schema_packages.general import (
    Program,
    Simulation,
)
from nomad_simulations.schema_packages.outputs import TotalEnergy, TotalForce

# nomad-parser-gsd
from nomad_parser_gsd.schema_packages.schema import (
    Author,
    ModelSystem,
    TrajectoryOutputs,
    CustomProperty,
    ParamEntry,
    ForceEntry,
    EnergyEntry,
    Stress,
)

logging = structlog.get_logger()
try:
    import gsd.fl as gsdfl
    import gsd.hoomd as gsdhoomd
    from gsd.hoomd import HOOMDTrajectory
    # import gsd.pygsd as gsdpy
except ImportError:
    logging.warning('Required module gsd.hoomd not found.')
    gsdhoomd = False
    # gsdpy = False

configuration = config.get_plugin_entry_point(
    'nomad_parser_gsd.parsers:parser_entry_point'
)

MOL = 6.022140857e23


class GSDFileParser(FileParser):
    def __init__(self):
        super().__init__(None)

    @property
    def filegsd(self):
        if self._file_handler is None:
            try:
                self._file_handler = gsdhoomd.open(name=self.mainfile, mode='r')
            except Exception:
                self.logger.error('Error reading gsd file.')

            if type(self._file_handler) is not HOOMDTrajectory:
                self.logger.error(
                    'Uknown GSD file object, only HOOMDTrajectory objects are supported.'
                )
        return self._file_handler

    def get_value(self, group, path: str, default=None):
        """
        Extracts group or dataset from group object based on path, and returns default if not defined.
        """
        section_segments = path.split('.')
        for section in section_segments:
            try:
                value = getattr(group, section)
                group = value
            except AttributeError:
                return

        return value if value is not None else default

    def parse(self, path: str = None, **kwargs):
        frame_path = '.'.join(path.split('.')[1:])
        frame = kwargs.get('frame', None)

        if not frame:
            value = None
        else:
            value = self.get_value(frame, frame_path)

        self._results[path] = value


class GSDParser(MDParser):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._data_parser = GSDFileParser()
        self._maindir = None
        self._gsd_files = None
        self._basename = None
        self._program_dict = None
        self._n_frames = None
        self._first_frame = True
        # ! Test to figure out what format is expected for the particles group:
        self.h5_write = h5py.File('test_gsd-nomad.h5', 'w')
        self._time_unit = ureg.picosecond
        # ['N', 'position', 'orientation', 'types', 'typeid', 'mass', 'charge',
        # 'diameter', 'body', 'moment_inertia', 'velocity', 'angmom', 'image',
        # 'type_shapes']
        self._nomad_to_particles_group_map = {
            'n_atoms': 'N',
            'positions': 'position',
            'velocities': 'velocity',
            'forces': None,
            'labels': 'types',
            'label': None,
            'mass': 'mass',
            'charge': 'charge',
        }
        self._nomad_to_box_group_map = {
            'step': 'step',
            'lattice_vectors': '_box',
            'periodic_boundary_conditions': None,
            'dimensionality': 'dimensions',
        }

    # Load GSD file as file layer object to access generating program name and version.
    def get_program_info(self):
        _file_layer = gsdfl.open(name=self.mainfile, mode='r')
        _application = _file_layer.application.split(' ')

        _program_info = dict()

        _program_info['name'] = _application[0]
        _program_info['version'] = _application[1]
        _program_info['schema'] = (
            _file_layer.schema
        )  # Name of the data schema. In example: 'hoomd'
        _program_info['schema_version'] = (
            _file_layer.schema_version
        )  # TODO: tuple(major, minor). If keep, format to major.minor?
        _program_info['gsd_version'] = (
            _file_layer.gsd_version
        )  # TODO: tuple(major, minor). Format?
        _program_info['gsd_author_name'] = None
        _program_info['gsd_author_email'] = None
        _program_info['gsd_creator_name'] = 'Sharon Glotzer'
        _program_info['gsd_creator_version'] = (
            'https://glotzerlab.engin.umich.edu/hoomd-blue/'
        )

        self._n_frames = _file_layer.nframes  # ? Not ideal to access in get_program, but only file layer object has attribute nframes

        _file_layer.close()

        return _program_info

    def get_molecules_from_bond_list(
        self,
        n_particles: int,
        bond_list: List[int],
        particle_types: List[str] = None,
        particles_typeid=None,
    ):
        """
        Returns a dictionary with molecule info from the list of bonds
        """

        import networkx

        system_graph = networkx.empty_graph(n_particles)
        system_graph.add_edges_from([(i[0], i[1]) for i in bond_list])
        molecules = [
            system_graph.subgraph(c).copy()
            for c in networkx.connected_components(system_graph)
        ]

        def get_composition(children_names):
            """
            Given a list of children, return a compositional formula as a function of
            these children. The format is <child_1>(n_child_1)<child_2>(n_child_2)...
            """
            children_count_tup = np.unique(children_names, return_counts=True)
            formula = ''.join(
                [f'{name}({count})' for name, count in zip(*children_count_tup)]
            )
            return formula

        mol_dict = []
        for i_mol, mol in enumerate(molecules):
            mol_dict.append({})
            mol_dict[i_mol]['indices'] = np.array(mol.nodes(), dtype=np.int32)
            mol_dict[i_mol]['bonds'] = np.array(mol.edges(), dtype=np.int32)
            mol_dict[i_mol]['type'] = 'molecule'
            mol_dict[i_mol]['is_molecule'] = True
            if particles_typeid is None and len(particle_types) == n_particles:
                mol_dict[i_mol]['names'] = [
                    particle_types[int(x)]
                    for x in sorted(np.array(mol_dict[i_mol]['indices']))
                ]
            if particle_types is not None and particles_typeid is not None:
                mol_dict[i_mol]['names'] = [
                    particle_types[particles_typeid[int(x)]]
                    for x in sorted(np.array(mol_dict[i_mol]['indices']))
                ]
            mol_dict[i_mol]['formula'] = get_composition(mol_dict[i_mol]['names'])

        return mol_dict

    def is_same_molecule(self, mol_1: dict, mol_2: dict):
        """
        Checks whether the 2 input molecule dictionaries represent the same
        molecule type, i.e., same particle types and corresponding bond connections.
        """

        if sorted(mol_1['names']) == sorted(mol_2['names']):
            mol_1_shift = np.min(mol_1['indices'])
            mol_2_shift = np.min(mol_2['indices'])
            mol_1_bonds_shift = mol_1['bonds'] - mol_1_shift
            mol_2_bonds_shift = mol_2['bonds'] - mol_2_shift

            bond_list_1 = [
                sorted((mol_1['names'][i], mol_1['names'][j]))
                for i, j in mol_1_bonds_shift
            ]
            bond_list_2 = [
                sorted((mol_2['names'][i], mol_2['names'][j]))
                for i, j in mol_2_bonds_shift
            ]

            bond_list_names_1, bond_list_counts_1 = np.unique(
                bond_list_1, axis=0, return_counts=True
            )
            bond_list_names_2, bond_list_counts_2 = np.unique(
                bond_list_2, axis=0, return_counts=True
            )

            bond_list_dict_1 = {
                bond[0] + '-' + bond[1]: bond_list_counts_1[i_bond]
                for i_bond, bond in enumerate(bond_list_names_1)
            }
            bond_list_dict_2 = {
                bond[0] + '-' + bond[1]: bond_list_counts_2[i_bond]
                for i_bond, bond in enumerate(bond_list_names_2)
            }
            if bond_list_dict_1 == bond_list_dict_2:
                return True

            return False

        return False

    def get_particles_group(self, bond_list: np.array, molecule_labels=None):
        n_atoms = self._particle_data_dict['N']
        particle_types = [
            self._particle_data_dict['types'][typeid]
            for idx, typeid in enumerate(self._particle_data_dict['typeid'])
        ]
        molecules = self.get_molecules_from_bond_list(
            n_atoms,
            bond_list,
            particle_types=particle_types,
            particles_typeid=None,
        )
        # create the topology
        mol_groups = []
        mol_groups.append({})
        mol_groups[0]['molecules'] = []
        mol_groups[0]['molecules'].append(molecules[0])
        mol_groups[0]['type'] = 'molecule_group'
        mol_groups[0]['is_molecule'] = False
        print(mol_groups)
        for mol in molecules[1:]:
            flag_mol_group_exists = False
            for i_mol_group in range(len(mol_groups)):
                if self.is_same_molecule(mol, mol_groups[i_mol_group]['molecules'][0]):
                    mol_groups[i_mol_group]['molecules'].append(mol)
                    flag_mol_group_exists = True
                    break
            if not flag_mol_group_exists:
                mol_groups.append({})
                mol_groups[-1]['molecules'] = []
                mol_groups[-1]['molecules'].append(mol)
                mol_groups[-1]['type'] = 'molecule_group'
                mol_groups[-1]['is_molecule'] = False

        if not molecule_labels:
            molecule_labels = [f'mol_type_{idx}' for idx in range(len(mol_groups))]

        for i_mol_group, mol_group in enumerate(mol_groups):
            mol_groups[i_mol_group]['formula'] = (
                molecule_labels[i_mol_group]
                + '('
                + str(len(mol_group['molecules']))
                + ')'
            )
            mol_groups[i_mol_group]['label'] = 'group_' + str(
                molecule_labels[i_mol_group]
            )
            mol_group_indices = []
            for i_molecule, molecule in enumerate(mol_group['molecules']):
                molecule['label'] = molecule_labels[i_mol_group]
                mol_indices = molecule['indices']
                mol_group_indices.append(mol_indices)
            mol_group['indices'] = np.concatenate(mol_group_indices)

        topology_keys = [
            'type',
            'formula',
            'particles_group',
            'label',
            'is_molecule',
            'indices',
        ]
        custom_keys = ['molecules', 'residue_groups', 'residues']  # ? Remove residues?
        mol_groups_dict, topology_dict = dict(), dict()
        mol_groups_label = 'particles_group'  # ? Need?
        for i_mol_group, mol_group in enumerate(
            mol_groups
        ):  # mol_group: dict of same molecule type
            gsd_mol_group_label = f'group_{molecule_labels[i_mol_group]}'
            gsd_mol_group_dict = dict()
            for mol_group_key in mol_group.keys():
                if mol_group_key not in topology_keys + custom_keys:
                    continue
                if mol_group_key != 'molecules':
                    gsd_mol_group_dict[mol_group_key] = mol_group[mol_group_key]
                else:
                    gsd_molecules_group_label = 'particles_group'
                    gsd_molecules_group_dict = dict()
                    for i_molecule, molecule in enumerate(mol_group[mol_group_key]):
                        gsd_molecule_label = f'molecule_{str(i_molecule)}'
                        gsd_molecule_dict = dict()
                        for mol_key in molecule.keys():
                            if mol_key not in topology_keys + custom_keys:
                                continue
                            # if mol_key != 'residue_groups':
                            gsd_molecule_dict[mol_key] = molecule[mol_key]
                        gsd_molecules_group_dict[gsd_molecule_label] = gsd_molecule_dict
                    gsd_mol_group_dict[gsd_molecules_group_label] = (
                        gsd_molecules_group_dict
                    )
            mol_groups_dict[gsd_mol_group_label] = gsd_mol_group_dict
        topology_dict[mol_groups_label] = mol_groups_dict
        print(topology_dict.keys())
        for key in topology_dict.keys():
            print(topology_dict[key].keys())
            for key2 in topology_dict[key].keys():
                print(topology_dict[key][key2].keys())

        return topology_dict  # mol_groups

    def get_connectivity(self, interactions, molecule_labels=None):
        # TODO: Carry molecule_labels through to a meaningful point in case molecule labels were passed by user.
        _connectivity = dict()
        for key in interactions.keys():
            if interactions[key]['N'] == 0:
                self.logger.warn(f'No {key} information found in GSD file.')
                _connectivity[key] = []  # None
            else:
                _connectivity[key] = list(
                    map(tuple, interactions[key]['group'].tolist())
                )

        _connectivity['particles_group'] = self.get_particles_group(
            interactions['bonds']['group'], molecule_labels=molecule_labels
        )

        return _connectivity

    def get_system_info(self, frame_idx=None, frame=None):
        self._system_info = {'system': dict(), 'outputs': dict()}
        _path = f'{frame_idx}'
        _interaction_types = [
            'bonds',
            'angles',
            'dihedrals',
            'impropers',
            'constraints',
            'pairs',
        ]

        def get_particle_parameters(path: str = None, frame=None):
            n_particles = self._data_parser.get(f'{path}.N', frame=frame)
            if n_particles is None:
                return dict()
            else:
                return self._data_parser.get(path, frame=frame).__dict__

        self._particle_data_dict = get_particle_parameters(
            path=f'{frame_idx}.particles', frame=frame
        )

        if self._particle_data_dict is None:
            self.logger.warning(
                f'No number of particles available in frame {frame_idx}. Other'
                ' particle attributes will not be stored for frame {frame_idx}.'
            )

        def get_value(value, path=None, frame=None):
            if value is None:
                return value
            try:
                value = self._data_parser.get(
                    f'{path}.{value}' if path else value, frame=frame
                ).__dict__
                if value is None:
                    self.logger.warning(
                        f'No attributes found for key {value}. {value.upper()} attributes will '
                        'not be stored.'
                    )
                    return None
                else:
                    return value
            except KeyError:
                return value
            # TODO: handle data chunks that are not standard objects
            except AttributeError:
                pass

        info_keys = {
            'step': ['system', 'outputs'],
            'n_atoms': 'system',
            'positions': 'system',
            'labels': 'system',
            'mass': 'system',
            'velocities': 'system',
            'charge': 'system',
            'lattice_vectors': 'system',
            'periodic_boundary_conditions': 'system',
            'dimensionality': 'system',
            'forces': 'outputs',
            'label': 'outputs',
        }

        # Get quantities from particles chunk of GSD file
        for key, gsd_key in self._nomad_to_particles_group_map.items():
            section = info_keys[key]
            if isinstance(section, list):
                for sec in section:
                    self._system_info[sec][key] = (
                        self._particle_data_dict[gsd_key]
                        if gsd_key is not None
                        else None
                    )
            else:
                self._system_info[section][key] = (
                    self._particle_data_dict[gsd_key] if gsd_key is not None else None
                )

        def box_to_matrix_converter(box):
            """
            Converts the lattice vectors from lengths and angles to a triclinic matrix
            expected by Nomad.
            Definition: https://hoomd-blue.readthedocs.io/en/v4.8.2/package-hoomd.html#hoomd.Box
            - a_1 is parallel to the unit vector e_x = (1, 0, 0)
            - xy describes the tilt of a_2 with respect to a_1
            - xz and yz describe the tilt of a_3 with respect to a_1 and a_2
            """

            def invalid_angles(xy, xz, yz):
                """
                The volume discriminant of a triclinic box is only negative or zero
                for triplets of box angles that lead to an invalid box shape
                (i.e., the sum of any two angles is less than or equal to the third).
                """

                # Calculate cosines of the angles
                cos_gamma = xy / np.sqrt(1 + np.power(xy, 2))
                cos_beta = xz / np.sqrt(1 + np.power(xz, 2) + np.power(yz, 2))
                cos_alpha = (xy * xz + yz) / (
                    np.sqrt(1 + np.power(xy, 2))
                    * np.sqrt(1 + np.power(xz, 2) + np.power(yz, 2))
                )

                # Convert cosines to angles in degrees
                gamma = np.degrees(np.arccos(cos_gamma))
                beta = np.degrees(np.arccos(cos_beta))
                alpha = np.degrees(np.arccos(cos_alpha))

                return not (
                    (alpha + beta > gamma)
                    or (alpha + gamma > beta)
                    or (beta + gamma > alpha)
                )

            if not (
                np.all(box[:2] > 0.0)
                and box[2] >= 0.0  # Third length can be zero for 2D systems
            ):
                # invalid box dimensions, return zero vectors:
                self.logger.warning(
                    f'Invalid box lengths: Required Lx, Ly > 0.0 and Lz >= 0.0. \
                        Found Lx = {box[0]}, Ly = {box[1]}, Lz = {box[2]}.'
                )
                return np.zeros((3, 3), dtype=np.float64)
            elif box[3] == box[4] == box[5] == 0.0:
                # box is orthogonal, return a diagonal matrix:
                return np.diag(box[:3].astype(np.float64, copy=False))
            else:
                # box is triclinic, convert to matrix:
                Lx, Ly, Lz, xy, xz, yz = box[0], box[1], box[2], box[3], box[4], box[5]
                box = np.zeros((3, 3), dtype=np.float64)

                if invalid_angles(xy, xz, yz):
                    # invalid box shape, return zero vectors:
                    self.logger.warning(
                        f'The tilt factors xy = {xy}, xz = {xz}, yz = {yz} do \
                            not satisfy the volume condition for a triclinic box.'
                    )
                    return np.zeros((3, 3), dtype=np.float64)

                box[0, 0] = Lx
                box[1, 0] = xy * Ly
                box[1, 1] = Ly
                box[2, 0] = xz * Lz
                box[2, 1] = yz * Lz
                box[2, 2] = Lz

                return box.astype(np.float64, copy=False)

        # Get step and box attributes from configurations chunk of GSD file:
        for key, gsd_key in self._nomad_to_box_group_map.items():
            section = info_keys[key]
            _values_dict = get_value('configuration', path=_path, frame=frame)
            if gsd_key == '_box':
                _values_dict[gsd_key] = box_to_matrix_converter(_values_dict[gsd_key])
            if isinstance(section, list):
                for sec in section:
                    self._system_info[sec][key] = (
                        _values_dict[gsd_key] if gsd_key is not None else None
                    )
            else:
                self._system_info[section][key] = (
                    _values_dict[gsd_key] if gsd_key is not None else None
                )

        # Extract interacton infromation from frame,
        # build connectivity structure following Nomad-H5MD schema.
        _interaction_dicts = dict()
        for interaction in _interaction_types:
            _interaction_dicts[interaction] = get_value(
                interaction, path=_path, frame=frame
            )
        self._connectivity = self.get_connectivity(_interaction_dicts)

        return self._system_info

    def parse_system_hierarchy(
        self,
        nomad_sec: ModelSystem,
        gsd_sec_particlesgroup: dict,
        # path_particlesgroup='',
    ):
        data = {}
        for key in gsd_sec_particlesgroup.keys():
            print('gsd_sec_particlesgroup key:', key)
            particles_group = {
                group_key: gsd_sec_particlesgroup[key].get(group_key, {})
                for group_key in gsd_sec_particlesgroup[key].keys()
            }
            print('particles_group keys:', particles_group.keys())
            sec_model_system = ModelSystem()
            nomad_sec.model_system.append(sec_model_system)
            data['branch_label'] = particles_group.pop('label', None)
            data['atom_indices'] = particles_group.pop('indices', None)
            # TODO remove the deprecated below from the test file
            # sec_atomsgroup.type = particles_group.pop("type", None) #? deprecate?
            particles_group.pop('type', None)
            # sec_atomsgroup.is_molecule = particles_group.pop("is_molecule", None) #? deprecate?
            particles_group.pop('is_molecule', None)
            particles_group.pop('formula', None)  # covered in normalization now
            # write all the standard quantities to the archive
            self.parse_section(data, sec_model_system)
            particles_subgroup = particles_group.pop('particles_group', None)
            # set the remaining attributes
            for particles_group_key in particles_group.keys():
                val = particles_group.get(particles_group_key)
                # sec_model_system.custom_system_attributes.append(
                #     # ! As long as value is dictionary, use SubSection
                #     # ParamEntry(name=particles_group_key, value=val, unit=units)
                # )

            # get the next branch level
            if particles_subgroup:
                self.parse_system_hierarchy(
                    sec_model_system,
                    particles_subgroup,
                    # f'{path_particlesgroup_key}.particles_group',
                )

    def parse_system(self, simulation, frame_idx=None, frame=None):
        particles_dict = self._system_info.get('system')
        # print(particles_dict.keys())
        _path = f'{frame_idx}'
        if not particles_dict:
            self.logger.error('No particle information found in GSD file.')
            return

        self._system_time_map = {}  # ? Is this required? What is ist used for?

        # TODO: extend to support visualization of time-dependent bond lists and topologies from (semi) grand canonical ensembles.
        if self._first_frame is True:
            self.logger.warning(
                'Only the topology of the first frame will be stored, '
                'grand canonical simulations are currently not supported.'
            )
            particles_dict['is_representative'] = True
            self._first_frame = False
        else:
            particles_dict['is_representative'] = False

        topology = self._connectivity['particles_group']
        particles_dict['labels'] = particles_dict.get('labels')

        bond_dict = self._data_parser.get(f'{frame_idx}.bonds', frame=frame).__dict__
        particles_dict['bond_list'] = bond_dict['group']

        # ! Natively, no time step stored in GSD file. Copy frame index instead,
        # ! alert user to missing information.
        time = particles_dict.pop('step')
        time_unit = time.units if hasattr(time, 'units') else None
        particles_dict['time_step'] = time.magnitude if time_unit is not None else time
        if time_unit is None:
            self.logger.warning(
                'No magnitude and unit information provided for the '
                'simulation time step'
            )

        # REMAP some of the data for the schema
        particles_dict['branch_label'] = f'System {time}'
        particle_cell_keys = [
            'n_atoms',
            'lattice_vectors',
            'periodic_boundary_conditions',
            'positions',
            'velocities',
            'labels',
        ]
        particles_dict['particle_cell'] = {}
        for key in particle_cell_keys:
            # HOOMD-Blue assumes periodic boundary conditions by default:
            # https://hoomd-blue.readthedocs.io/en/v4.8.2/package-hoomd.html#hoomd.Box.periodic
            if key == 'periodic_boundary_conditions':
                particles_dict.pop(key)
                particles_dict['particle_cell'][key] = [True] * 3
            else:
                particles_dict['particle_cell'][key] = particles_dict.pop(key)

        # ! MDParser.parse_trajectory_step doesn't work for the new schema, extending function.
        self.parse_trajectory_step(particles_dict, simulation)

        # TODO: parse and store topology in every step to accomodate time-dependent topologies
        if topology:
            self.parse_system_hierarchy(simulation.model_system[-1], topology)

    # TODO: Additional data are stored in the log dictionary as numpy arrays:
    def parse_outputs(self, simulation: Simulation):
        """
        Logged data encompasses values computed at simulation time that are too expensive
        or cumbersome to re-compute in post processing. This specification does not define
        specific chunk names or define logged data. Users may select any valid name for
        logged data chunks as appropriate for their workflow.
        """

        def get_logged_info(self):
            try:
                return gsdhoomd.read_log(name=self.mainfile, scalar_only=False)
            except FileNotFoundError:
                self.logger.warning(
                    'No additional logged data found, no user-defined data will be stored.'
                )
                return dict()

        # TODO: parse log dictionary, programmatically write to archive. Sections identified by dictionary keys.
        # ? Is the logged data section present in every frame, or only the last one?
        # ? Information on step(s) present or needs grabbing from upstream?

    def write_to_archive(self) -> None:
        #######################################################################
        # Access simulation file(s).
        #######################################################################
        self._maindir = os.path.dirname(
            self.mainfile
        )  # ? GSD output single file or more?
        self._gsd_files = [
            _file for _file in os.listdir(self._maindir) if _file.endswith('.gsd')
        ]
        self._basename = os.path.basename(self.mainfile).rsplit('.', 1)[0]
        self._data_parser.mainfile = self.mainfile
        if self._data_parser.filegsd is None:
            self.logger.warning('GSD file missing in GSD Parser.')
            return

        #######################################################################
        # Start populating NOMAD-simulations schema
        #######################################################################
        simulation = Simulation()
        self._program_dict = self.get_program_info()
        simulation.program = Program(
            name=self._program_dict['name'],
            version=self._program_dict['version'],
        )
        simulation.x_gsd_version = self._program_dict['gsd_version']
        simulation.x_gsd_schema = Program(
            name=self._program_dict.get('schema'),
            version=f"{self._program_dict.get('schema_version')[0]}.{self._program_dict.get('schema_version')[1]}",
        )
        simulation.x_gsd_author = Author(
            name=self._program_dict.get('gsd_author_name'),
            email=self._program_dict.get('gsd_author_email'),
        )
        simulation.x_gsd_creator = Program(
            name=self._program_dict.get('gsd_creator_name'),
            version=self._program_dict.get('gsd_creator_version'),
        )
        for frame_idx, frame in enumerate(self._data_parser.filegsd):
            self.get_system_info(frame_idx=frame_idx, frame=frame)
            self.parse_system(simulation, frame_idx=frame_idx, frame=frame)

            # ? Forces etc. are user-defined and read via get_logged_info
            # TODO: Extract observables from logged data, parse to ModelOutput
            # self.parse_outputs(simulation, frame_idx=frame_idx, frame=frame)

            # observable_keys = {
            #     'forces': 'calculation',
            # }
        # ! 'NameError: name 'EntryArchive' is not defined' because of TYPE_CHECKING=False?!
        self.archive.m_add_sub_section(EntryArchive.data, simulation)
