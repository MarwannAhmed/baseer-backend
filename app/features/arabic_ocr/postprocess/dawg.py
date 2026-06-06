import pickle
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DawgNode:
    children: dict = field(default_factory=dict)
    is_end:   bool = False


def build_dawg(word_list):

    root = DawgNode()
    for word in word_list:
        node = root
        for ch in word.strip():
            if ch not in node.children:
                node.children[ch] = DawgNode()
            node = node.children[ch]
        node.is_end = True
    return root


def dawg_search(root: DawgNode, prefix):
    node = root
    for ch in prefix:
        if ch not in node.children:
            return []
        node = node.children[ch]
    completions: list[str] = []
    _collect(node, prefix, completions)
    return completions

def _collect(node: DawgNode, current, results):
    if node.is_end:
        results.append(current)
    for ch, child in node.children.items():
        _collect(child, current + ch, results)

def save_dawg(root: DawgNode, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(root, f)

def load_dawg(path) -> DawgNode:
    with open(path, "rb") as f:
        return pickle.load(f)
