import json
from collections import deque
from typing import Any, Dict, List

import pandas as pd
from bs4 import BeautifulSoup, Comment, Tag


def get_path(element: Tag) -> str:
    """Generates an XPath-like DOM structure path."""
    path_parts = []

    for parent in element.parents:
        if parent.name is None:
            break

        siblings_of_same_tag = [s for s in parent.children if s.name == element.name and not isinstance(s, Comment)]

        try:
            index = siblings_of_same_tag.index(element)

            if len(siblings_of_same_tag) == 1:
                part = element.name
            else:
                part = f"{element.name}[{index}]"

        except ValueError:
            part = element.name

        path_parts.append(part)
        element = parent

    return "/".join(reversed(path_parts))


def extract_content(element: Tag, dom_path: str, path_to_id_map: Dict[str, int]) -> str:
    """Extracts direct text content mixed with ID reference for child tags, with escaping."""
    content_parts = []

    for content in element.contents:
        if isinstance(content, Comment):
            continue

        if isinstance(content, str):
            stripped_text = content.strip()
            if stripped_text:
                escaped_text = stripped_text.replace('{', r'\{').replace('}', r'\}')
                content_parts.append(escaped_text.replace('\n', ' '))

        elif isinstance(content, Tag):
            child_dom_path = get_path(content)
            child_id = path_to_id_map.get(child_dom_path)
            if child_id is not None:
                hierarchy_part = f"{{id{child_id}}}"
                content_parts.append(hierarchy_part)
            else:
                content_parts.append("[ID_NOT_FOUND]")

    if content_parts:
        return " ".join(content_parts)
    else:
        return None


def extract_dom_data(html_content: str) -> List[Dict[str, Any]]:
    """
    Performs DFS on the DOM starting from the <html> root and returns a list of dictionaries.
    The process is split into two passes: 1. Mapping paths to IDs, 2. Generating content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    data_list = []

    start_element = soup.html
    if not start_element:
        return data_list
    temp_data = []
    path_to_id_map: Dict[str, int] = {}
    element_counter = 0
    stack = [start_element]
    visited_elements = set()
    while stack:
        current_element = stack.pop()

        if current_element in visited_elements:
            continue
        visited_elements.add(current_element)

        dom_path = get_path(current_element)
        path_to_id_map[dom_path] = element_counter

        temp_data.append({
            "element": current_element,
            "id": element_counter,
            "xpath": dom_path,
            "attributes": json.dumps(current_element.attrs, ensure_ascii=False)
        })

        children = [child for child in current_element.children if isinstance(child, Tag)]
        for child in reversed(children):
            stack.append(child)

        element_counter += 1

    for item in temp_data:
        current_element = item["element"]
        dom_path = item["xpath"]
        content_output = extract_content(current_element, dom_path, path_to_id_map)
        final_item = {
            "id": item["id"],
            "xpath": item["xpath"],
            "attributes": item["attributes"],
            "contents": content_output
        }
        data_list.append(final_item)

    return data_list


if __name__ == '__main__':
    for i in range(30):
        with open(f"raw/{i}.html") as f:
            html = f.read()
        output_filename = f"data/{i}.csv"
        dom_array = extract_dom_data(html)
        df = pd.DataFrame(dom_array)
        df.to_csv(output_filename, index=False, encoding='utf-8')
