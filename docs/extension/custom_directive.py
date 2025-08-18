import os
import re
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

class TableInclude(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'table': str
    }

    def run(self):
        # Get the file path from the first argument
        file_path = self.arguments[0]

        # Get the environment to resolve the full path
        env = self.state.document.settings.env
        src_dir = os.path.abspath(env.srcdir)
        full_file_path = os.path.join(src_dir, file_path)

        # Check if the file exists
        if not os.path.exists(full_file_path):
            raise self.error(f"RST file {full_file_path} does not exist.")

        # Read the entire file content
        with open(full_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all tables with named targets
        table_pattern = r'(?:^\.\.\ _(.+?):\n)(.. list-table::.*?(?:\n\s*\*\s*-.*?)+)(?=\n\n|\Z)'
        table_matches = list(re.finditer(table_pattern, content, re.MULTILINE | re.DOTALL))

        # Get the specific table name from options
        table_name = self.options.get('table')

        # If no table specified, merge compatible tables
        if not table_name:
            raise self.error("The ':table:' option is required to specify which table to include.")

        # Find the specific table
        matching_tables = [
            match for match in table_matches
            if match.group(1).strip() == table_name
        ]

        if not matching_tables:
            raise self.error(f"Table '{table_name}' not found in {full_file_path}")

        # Extract the matched table content
        table_content = matching_tables[0].group(2)

        # Insert the table content into the current document
        self.state_machine.insert_input(table_content.splitlines(), full_file_path)
        return []

def setup(app):
    app.add_directive('include-table', TableInclude)
