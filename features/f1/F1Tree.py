from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class F1Sequence:
    nodes: List["F1Node"]
    start: int
    end: int


@dataclass
class F1Node:
    token: str
    start: int
    end: int
    children: List[F1Sequence]


F1Tree = Union[F1Node, F1Sequence]


class F1Parser:
    def __init__(self, text: str):
        self.text = text
        self.n = len(text)
        self.i = 0

    def parse(self) -> Optional[F1Sequence]:
        self._skip_ws()
        if self.i >= self.n:
            return None

        sequence = self._parse_sequence(stop_chars=set())
        if sequence is None:
            return None

        self._skip_ws()
        if self.i != self.n:
            return None
        return sequence

    def _skip_ws(self):
        while self.i < self.n and self.text[self.i].isspace():
            self.i += 1

    def _parse_sequence(self, stop_chars) -> Optional[F1Sequence]:
        self._skip_ws()
        nodes: List[F1Node] = []

        while self.i < self.n:
            self._skip_ws()
            if self.i >= self.n or self.text[self.i] in stop_chars:
                break

            node = self._parse_node()
            if node is None:
                return None
            nodes.append(node)

        if not nodes:
            return None

        return F1Sequence(nodes=nodes, start=nodes[0].start, end=nodes[-1].end)

    def _parse_node(self) -> Optional[F1Node]:
        self._skip_ws()
        if self.i >= self.n:
            return None

        start = self.i
        token, has_x = self._parse_unit_token()
        if not token:
            return None

        self._skip_ws()
        children: List[F1Sequence] = []

        if self.i < self.n and self.text[self.i] == '(':
            children = self._parse_children()
            if children is None:
                return None
        elif not has_x:
            # Modifier-only tails like "X(...)rR" are not valid tree nodes.
            return None

        return F1Node(token=token, start=start, end=self.i, children=children)

    def _parse_children(self) -> Optional[List[F1Sequence]]:
        if self.i >= self.n or self.text[self.i] != '(':
            return None

        self.i += 1
        self._skip_ws()

        if self.i < self.n and self.text[self.i] == ')':
            self.i += 1
            return []

        children: List[F1Sequence] = []
        while True:
            child = self._parse_sequence(stop_chars={',', ')'})
            if child is None:
                return None
            children.append(child)

            self._skip_ws()
            if self.i >= self.n:
                return None
            if self.text[self.i] == ',':
                self.i += 1
                self._skip_ws()
                continue
            if self.text[self.i] == ')':
                self.i += 1
                break
            return None

        return children

    def _parse_unit_token(self):
        token_start = self.i
        bracket_depth = 0
        has_x = False

        while self.i < self.n:
            ch = self.text[self.i]

            if bracket_depth == 0 and (ch.isspace() or ch in '(),'):
                break

            self.i += 1

            if ch == '[':
                bracket_depth += 1
            elif ch == ']' and bracket_depth > 0:
                bracket_depth -= 1
            elif bracket_depth == 0 and ch == 'X':
                has_x = True
                self._consume_attached_brackets()
                break

        return self.text[token_start:self.i], has_x

    def _consume_attached_brackets(self):
        while self.i < self.n and self.text[self.i] == '[':
            bracket_depth = 0
            while self.i < self.n:
                ch = self.text[self.i]
                self.i += 1
                if ch == '[':
                    bracket_depth += 1
                elif ch == ']':
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        break


def serialize_f1(tree: F1Tree) -> str:
    if isinstance(tree, F1Sequence):
        return ''.join(serialize_f1(node) for node in tree.nodes)

    if not tree.children:
        return tree.token

    return tree.token + '(' + ','.join(serialize_f1(child) for child in tree.children) + ')'


def collect_subtrees(tree: F1Tree) -> List[F1Tree]:
    out: List[F1Tree] = []

    if isinstance(tree, F1Sequence):
        if len(tree.nodes) > 1:
            out.append(tree)
        for node in tree.nodes:
            out.extend(collect_subtrees(node))
        return out

    out.append(tree)
    for child in tree.children:
        out.extend(collect_subtrees(child))
    return out
