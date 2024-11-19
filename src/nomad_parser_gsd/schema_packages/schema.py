#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
import nomad_simulations
import numpy as np
from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import (
    Context,
    MEnum,
    Quantity,
    Section,
    SectionProxy,
    SubSection,
    JSON,
)


class ParamEntry(ArchiveSection):
    """
    Generic section defining a parameter name and value
    """

    name = Quantity(
        type=str,
        shape=[],
        description="""
        Name of the parameter.
        """,
    )

    value = Quantity(
        type=JSON,
        shape=[],
        description="""
        Value of the parameter as a string.
        """,
    )

    unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the parameter as a string.
        """,
    )

    description = Quantity(
        type=str,
        shape=[],
        description="""
        Further description of the attribute.
        """,
    )


class CustomProperty(
    nomad_simulations.schema_packages.physical_property.PhysicalProperty
):
    """
    Section describing a general type of calculation.
    """

    value = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Value **magnitude** of the property. The unit is defined in the `unit` attribute.
        """,
    )

    unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the parameter as a string consistent with the UnitRegistry.pint module.
        """,
    )

    description = Quantity(
        type=str,
        shape=[],
        description="""
        Further description of the property.
        """,
    )


class EnergyEntry(ArchiveSection):
    """
    Section describing a general type of energy contribution.
    """

    name = Quantity(
        type=str,
        shape=[],
        description="""
        Name of the energy contribution.
        """,
    )

    value = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the energy contribution.
        """,
    )


class ForceEntry(ArchiveSection):
    """
    Section describing a general type of force contribution.
    """

    name = Quantity(
        type=str,
        shape=[],
        description="""
        Name of the force contribution.
        """,
    )

    value = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='newton',
        description="""
        Value of the force contribution.
        """,
    )


# class ForceCalculations(runschema.method.ForceCalculations):
#     m_def = Section(
#         validate=False,
#         extends_base_section=True,
#     )

#     x_h5md_parameters = SubSection(
#         sub_section=ParamEntry.m_def,
#         description="""
#         Contains non-normalized force calculation parameters.
#         """,
#         repeats=True,
#     )


# class NeighborSearching(runschema.method.NeighborSearching):
#     m_def = Section(
#         validate=False,
#         extends_base_section=True,
#     )

#     x_h5md_parameters = SubSection(
#         sub_section=ParamEntry.m_def,
#         description="""
#         Contains non-normalized neighbor searching parameters.
#         """,
#         repeats=True,
#     )


class ModelSystem(nomad_simulations.schema_packages.model_system.ModelSystem):
    """
    Model system used as an input for simulating the material.
    """

    custom_system_attributes = (
        SubSection(  # TODO should this be called parameters or attributes or what?
            sub_section=ParamEntry.m_def,
            description="""
        Contains additional information about the (sub)system .
        """,
            repeats=True,
        )
    )


class Stress(nomad_simulations.schema_packages.physical_property.PhysicalProperty):
    """ """

    value = Quantity(
        type=np.dtype(np.float64),
        unit='newton',
        description="""
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class TrajectoryOutputs(nomad_simulations.schema_packages.outputs.TrajectoryOutputs):
    m_def = Section(
        validate=False,
        extends_base_section=True,
    )

    x_h5md_custom_outputs = SubSection(
        sub_section=CustomProperty.m_def,
        description="""
        Contains other generic custom outputs that are not already defined.
        """,
        repeats=True,
    )


class Author(ArchiveSection):
    """
    Contains the specifications of the program.
    """

    name = Quantity(
        type=str,
        shape=[],
        description="""
        Specifies the name of the author who generated the h5md file.
        """,
    )

    email = Quantity(
        type=str,
        shape=[],
        description="""
        Author's email.
        """,
    )


class H5MDCreator(nomad_simulations.schema_packages.general.Program):
    """
    Contains the specifications of the program.
    """

    m_def = Section(
        validate=False,
        extends_base_section=True,
    )

    name = Quantity(
        type=str,
        shape=[],
        description="""
        Specifies the name of the author who generated the h5md file.
        """,
    )

    email = Quantity(
        type=str,
        shape=[],
        description="""
        Author's email.
        """,
    )


class Simulation(nomad_simulations.schema_packages.general.Simulation):
    m_def = Section(
        validate=False,
        extends_base_section=True,
    )

    # TODO Not sure how we are dealing with versioning with H5MD-NOMAD
    x_h5md_version = Quantity(
        type=np.dtype(np.int32),
        shape=[2],
        description="""
        Specifies the version of the h5md schema being followed.
        """,
    )

    x_h5md_author = SubSection(sub_section=Author.m_def)

    x_h5md_creator = SubSection(
        sub_section=nomad_simulations.schema_packages.general.Program.m_def
    )
