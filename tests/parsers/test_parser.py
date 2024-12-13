import logging
import os
import pytest
from unittest.mock import patch

from nomad.config import config
from nomad.datamodel import EntryArchive
from nomad_parser_gsd.parsers.parser import GSDParser


@pytest.fixture(scope='module', autouse=True)
def parser():
    return GSDParser()


# @pytest.fixture(scope='module')
# def file_path():
#     return (
#         '/home/bmohr/Documents/HOOMDblue_GSD_examples/KG_Sims/For_Nomad/trajectory.gsd'
#     )


# def test_plugin_entry_point():
#     """
#     Test that the plugin entry point is correctly configured.
#     """
#     entry_point_path = 'nomad_parser_gsd.parsers:parser_entry_point'

#     with patch.object(config, 'get_plugin_entry_point') as mock_get_plugin_entry_point:
#         # Mock the return value of the entry point function
#         mock_get_plugin_entry_point.return_value = 'mocked_parser_entry_point'
#         print(mock_get_plugin_entry_point.return_value)

#         # Call the function that retrieves the configuration
#         print(config, type(config))
#         configuration = config.get_plugin_entry_point(entry_point_path)

#         # Verify the entry point was resolved correctly
#         mock_get_plugin_entry_point.assert_called_once_with(entry_point_path)
#         assert configuration == 'mocked_parser_entry_point'


# def test_read_file():
#     parser = GSDParser()
#     assert (
#         parser._data_parser.filegsd(
#             '/home/bmohr/Documents/HOOMDblue_GSD_examples/KG_Sims/For_Nomad/trajectory.gsd'
#         )
#         is not None
#     )


def test_parse_file():
    archive = EntryArchive()
    # parser = GSDParser()
    parser.parse(
        '/home/bmohr/Documents/HOOMDblue_GSD_examples/KG_Sims/For_Nomad/trajectory.gsd',
        archive,
        logging.getLogger(),
    )


#     assert archive.workflow2.name == 'test'
