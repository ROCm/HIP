import os, re, sys
from typing import List, Optional


def is_checkbox(line: str) -> bool:
    return bool(re.match(r"^\s*-\s*\[[ xX]\]\s*.+", line))


def is_checked(line: str) -> bool:
    return bool(re.match(r"^\s*-\s*\[[xX]\]\s*.+", line))


def is_comment(line: str) -> bool:
    return bool(re.match(r"^\s*<!--.*-->\s*$", line))


def text_clean(lines: List[str]) -> str:
    text = [line for line in lines if not is_comment(line)]
    return "".join("".join(text).strip().split())


def validate_section(section_name: str, lines: List[str]) -> Optional[str]:
    has_checkboxes = any(is_checkbox(line) for line in lines)
    if has_checkboxes:
        if not any(is_checked(line) for line in lines):
            return f"Section {section_name} is a checklist without selections"
        return None
    if not text_clean(lines):
        return f"Section {section_name} is empty text section"
    return None


def check_description(description: str) -> List[str]:
    if not description:
        # pull_request_template is not merged yet, so treat as valid for now
        return []
        # return ["PR description is empty"]

    sections = []
    current_section = None
    current_lines = []
    errors = []

    for line in description.splitlines():
        header_match = re.match(r"^\s*##\s*(.+?)\s*$", line)
        if header_match:
            if current_section:
                sections.append((current_section, current_lines))
            current_section = header_match.group(1)
            current_lines = []
        elif current_section:
            current_lines.append(line)

    if current_section:
        sections.append((current_section, current_lines))

    if not sections:
        return ["No sections available, template is empty"]

    for section_name, section_lines in sections:
        error = validate_section(section_name, section_lines)
        if error:
            errors.append(error)

    return errors


if __name__ == "__main__":
    pr_description = os.getenv("PR_DESCRIPTION", "")

    errors = check_description(pr_description)
    if not errors:
        print("All good")
        exit(0)
    print("\n".join(errors))
    exit(1)
