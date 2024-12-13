from nomad.config.models.plugins import ParserEntryPoint
from pydantic import Field
from typing import Optional


class GSDParserEntryPoint(ParserEntryPoint):
    parser_class_name: str = Field(
        description="""
        The fully qualified name of the Python class that implements the parser.
        This class must have a function `def parse(self, mainfile, archive, logger)`.
    """
    )
    level: int = Field(
        0,
        description="""
        Order of execution of parser with respect to other parsers.
    """,
    )
    code_name: Optional[str]
    code_homepage: Optional[str]
    code_category: Optional[str]
    metadata: Optional[dict] = Field(
        description="""
        Metadata passed to the UI. Deprecated. """
    )

    def load(self):
        from nomad.parsing import MatchingParserInterface

        return MatchingParserInterface(**self.dict())


parser_entry_point = GSDParserEntryPoint(
    name='nomad-parser-gsd',
    aliases=['nomad-parser-gsd', 'gsd'],
    description='Entry point for the GSD parser. ',
    python_package='nomad-parser-gsd.parsers',
    parser_class_name='nomad_parser_gsd.parsers.parser.GSDParser',
    level=1,
    code_name='GSD',
    code_category='MD code',
    # mainfile_binary_header_re=b'0x65DF65DF65DF65DF',  #! Seems to be wrong, didn't cause issues initially.
    mainfile_name_re=r'^.*\.gsd$',
    metadata={
        'codeCategory': 'MD code',
        'codeLabel': 'GSD',
        'codeLabelStyle': 'All in capitals',
        'codeName': 'gsd',
        'parserGitUrl': 'https://github.com/FAIRmat-NFDI/nomad-parser-gsd.git',
        'parserSpecific': '',
        'preamble': '',
        # 'status': 'production',
        'tableOfFiles': '',
    },
)
