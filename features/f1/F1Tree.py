from dataclasses import dataclass
from typing import List, Optional


@dataclass
class F1Node:
    token: str
    start: int
    end: int
    children: List["F1Node"]


class F1Parser:
    def __init__(self, text: str):
        self.text = text
        self.n = len(text)
        self.i = 0

    def parse(self) -> Optional[F1Node]:
        self._skip_ws()
        if self.i >= self.n:
            return None

        node = self._parse_node()
        if node is None:
            return None

        self._skip_ws()
        if self.i != self.n:
            return None
        return node

    def _skip_ws(self):
        while self.i < self.n and self.text[self.i].isspace():
            self.i += 1

    def _parse_node(self) -> Optional[F1Node]:
        self._skip_ws()
        if self.i >= self.n:
            return None

        start = self.i
        token = self._parse_token()
        if not token:
            return None

        self._skip_ws()
        children: List[F1Node] = []

        if self.i < self.n and self.text[self.i] == '(':
            self.i += 1
            self._skip_ws()

            if self.i < self.n and self.text[self.i] == ')':
                self.i += 1
            else:
                while True:
                    child = self._parse_node()
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

        end = self.i
        return F1Node(token=token, start=start, end=end, children=children)

    def _parse_token(self) -> str:
        token_chars = []
        while self.i < self.n:
            ch = self.text[self.i]
            if ch.isspace() or ch in '(),':
                break
            token_chars.append(ch)
            self.i += 1
        return ''.join(token_chars)


def serialize_f1(node: F1Node) -> str:
    if not node.children:
        return node.token
    return node.token + '(' + ','.join(serialize_f1(child) for child in node.children) + ')'


def collect_subtrees(node: F1Node) -> List[F1Node]:
    out = [node]
    for child in node.children:
        out.extend(collect_subtrees(child))
    return out
