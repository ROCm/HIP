# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from rocm_docs import ROCmDocs
from typing import Any, Dict, List


name = "HIP"
version = "5.4.4"
if len(version) > 0:
    name = f"{name} {version}"

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(f"{name} Documentation")
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.enable_api_reference()
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# rocm-docs-core might or might not have changed these yet (depending on version),
# and we don't want to wipe their settings if they did
if not "html_theme_options" in globals():
    html_theme_options: Dict[str, Any] = {}
if not "exclude_patterns" in globals():
    exclude_patterns: List[str] = []

html_theme_options["show_navbar_depth"] = 2
exclude_patterns.append("doxygen/mainpage.md")
external_projects_current_project = "hip"
