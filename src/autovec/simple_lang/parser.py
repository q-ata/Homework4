import operator

from lark import Lark, Token, Tree

from ..codegen import NumpyBufferFType
from . import nodes as smpl
import numpy as np

nary_ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}


unary_ops = {
    "+": operator.pos,
    "-": operator.neg,
}


lark_parser = Lark(
    """
    %import common.CNAME
    %import common.SIGNED_INT
    %import common.NEWLINE
    %ignore " " | NEWLINE  // Disregard spaces in text
    %ignore COMMENT

    start: "function" FUNC_NAME "(" func_args ")" "->" "[" (NUM ",")* NUM? "]" ":" prgm "end"
    func_args: (var_defn ",")* var_defn?
    var_defn: VAR "[" (NUM ",")* NUM? "]"
    prgm: (assign | loop | return)*
    loop_body: (assign | loop)*
    loop_header: loop_vanilla | loop_strided
    loop_vanilla: LOOP_IDX "in" "range" "(" NUM "," NUM ")"
    loop_strided: LOOP_IDX "in" "range" "(" NUM "," NUM "," NUM ")"
    loop: "for" loop_header loop_body "end"
    return: "return" VAR

    assign: access "=" expr
    access: VAR "[" (id_expr ",")* id_expr? "]"

    // Array Expressions with operator precedence (lowest to highest)
    expr: add_expr
    add_expr: mul_expr ((PLUS | MINUS) mul_expr)*
    mul_expr: unary_expr ((MUL) unary_expr)*
    unary_expr: (PLUS | MINUS) unary_expr | primary
    primary: access | LITERAL | "(" expr ")"

    // Index Expressions with operator precedence (lowest to highest)
    id_expr: id_add_expr | id_vector | id_strided_vector
    id_vector: NUM ":" NUM
    id_strided_vector: NUM ":" NUM ":" NUM
    id_add_expr: id_mul_expr ((PLUS | MINUS) id_mul_expr)*
    id_mul_expr: id_unary_expr ((MUL) id_unary_expr)*
    id_unary_expr: (PLUS | MINUS) id_unary_expr | id_primary
    id_primary: LOOP_IDX | LITERAL | "(" expr ")"

    PLUS: "+"
    MINUS: "-"
    MUL: "*"

    COMMENT: "#" /[^\\n]*/
    LITERAL: SIGNED_INT
    NUM: /[0-9]+/
    LOOP_IDX: CNAME
    VAR: CNAME
    FUNC_NAME: CNAME
"""
)


class SyntaxError(Exception):
    pass


class SemanticError(Exception):
    pass


class SimpleLangParser:
    def __init__(self):
        self.var_map: dict[str, smpl.Variable] = {}

    def _parse_id_expr(self, t: Tree) -> smpl.SimpleLangExpression:
        match t:
            case Tree(
                "id_expr" | "id_add_expr" | "id_mul_expr" | "id_unary_expr",
                [child],
            ):
                return self._parse_id_expr(child)
            case Tree("id_vector", [Token("NUM", start), Token("NUM", end)]):
                return smpl.VectorIndex(int(start), int(end), 1)
            case Tree(
                "id_strided_vector",
                [Token("NUM", start), Token("NUM", end), Token("NUM", stride)],
            ):
                return smpl.VectorIndex(int(start), int(end), int(stride))
            case Tree(
                "id_add_expr" | "id_mul_expr",
                args,
            ) if (
                len(args) > 1
            ):
                expr = self._parse_id_expr(args[0])
                for i in range(1, len(args), 2):
                    arg = self._parse_id_expr(args[i + 1])
                    op = args[i].value
                    if op not in nary_ops:
                        raise ValueError(f"Expected nary op but got {op}")
                    expr = smpl.Call(smpl.Literal(nary_ops[op]), (expr, arg))
                return expr

            case Tree("id_unary_expr", [op, arg]):
                if op not in unary_ops:
                    raise ValueError(f"Expected unary op but got {op}")
                return smpl.Call(
                    smpl.Literal(unary_ops[op]), (self._parse_id_expr(arg),)
                )
            case Tree("id_primary", [Token("LOOP_IDX", idx)]):
                return smpl.Index(idx)
            case Tree("id_primary", [Token("LITERAL", val)]):
                return smpl.Literal(int(val))  # type: ignore[union-attr]
            case _:
                raise SyntaxError(
                    f"_parse_id_expr encountered unknown node during parsing: {type(t)}"
                )

    def _parse_expr(self, t: Tree) -> smpl.SimpleLangExpression:
        match t:
            case Tree(
                "expr" | "add_expr" | "mul_expr" | "unary_expr",
                [child],
            ):
                return self._parse_expr(child)
            case Tree(
                "add_expr" | "mul_expr",
                args,
            ) if (
                len(args) > 1
            ):
                expr = self._parse_expr(args[0])
                for i in range(1, len(args), 2):
                    arg = self._parse_expr(args[i + 1])
                    op = args[i].value
                    if op not in nary_ops:
                        raise ValueError(f"Expected nary op but got {op}")
                    expr = smpl.Call(smpl.Literal(nary_ops[op]), (expr, arg))
                return expr

            case Tree("unary_expr", [op, arg]):
                if op not in unary_ops:
                    raise ValueError(f"Expected unary op but got {op}")
                return smpl.Call(smpl.Literal(unary_ops[op]), (self._parse_expr(arg),))
            case Tree("primary", [Tree("access", [Token("VAR", var), *idxs])]):
                return smpl.Load(
                    self.var_map[var],  # type: ignore[union-attr]
                    tuple(self._parse_id_expr(idx) for idx in idxs),  # type: ignore[union-attr]
                )
            case Tree("primary", [Token("LITERAL", val)]):
                return smpl.Literal(int(val))  # type: ignore[union-attr]
            case _:
                raise SyntaxError(
                    f"_parse_expr encountered unknown node during parsing: {type(t)}"
                )

    def _parse_body(self, t: Tree) -> smpl.SimpleLangNode:
        match t:
            case Tree(
                "start",
                [
                    Token("FUNC_NAME", func_name),
                    Tree("func_args", args),
                    *return_args,
                    func_body_tree,
                ],
            ):
                return smpl.Function(
                    name=smpl.Variable(
                        func_name,
                        NumpyBufferFType(
                            np.float64, tuple([int(arg.value) for arg in return_args])
                        ),
                    ),
                    args=tuple([self._parse_body(arg) for arg in args]),
                    body=self._parse_body(func_body_tree),
                )

            case Tree("var_defn", [Token("VAR", var), *args]):
                if var in self.var_map:
                    raise SemanticError(f"Variable {var} is redefined.")
                self.var_map[var] = smpl.Variable(
                    var,
                    NumpyBufferFType(
                        np.float64, tuple([int(arg.value) for arg in args])
                    ),
                )
                return self.var_map[var]

            case Tree("prgm", prgm_nodes):
                return smpl.Block(
                    tuple([self._parse_body(node) for node in prgm_nodes])
                )

            case Tree(
                "loop",
                [
                    Tree(
                        "loop_header",
                        [
                            Tree(
                                "loop_vanilla",
                                [
                                    Token("LOOP_IDX", idx),
                                    Token("NUM", start),
                                    Token("NUM", stop),
                                ],
                            ),
                        ],
                    ),
                    Tree("loop_body", loop_body),
                ],
            ):
                return smpl.ForLoop(
                    smpl.Index(idx),
                    smpl.Literal(int(start)),
                    smpl.Literal(int(stop)),
                    smpl.Literal(1),
                    smpl.Block(tuple([self._parse_body(body) for body in loop_body])),
                )

            case Tree(
                "loop",
                [
                    Tree(
                        "loop_header",
                        [
                            Tree(
                                "loop_strided",
                                [
                                    Token("LOOP_IDX", idx),
                                    Token("NUM", start),
                                    Token("NUM", stop),
                                    Token("NUM", stride),
                                ],
                            ),
                        ],
                    ),
                    Tree("loop_body", loop_body),
                ],
            ):
                return smpl.ForLoop(
                    smpl.Index(idx),
                    smpl.Literal(int(start)),
                    smpl.Literal(int(stop)),
                    smpl.Literal(int(stride)),
                    smpl.Block(tuple([self._parse_body(body) for body in loop_body])),
                )

            case Tree("return", [Token("VAR", var)]):
                return smpl.Return(self.var_map[var])

            case Tree(
                "assign", [Tree("access", [Token("VAR", var), *idxs]), expr_node]
            ):
                if var not in self.var_map:
                    raise SemanticError(f"No variable named {var} was defined.")
                return smpl.Store(
                    self.var_map[var],
                    tuple(self._parse_id_expr(idx) for idx in idxs),
                    self._parse_expr(expr_node),
                )

            case _:
                raise SyntaxError(
                    f"_parse_body encountered unknown node during parsing: {type(t)}"
                )

    # TODO: Add checks to ensure operations happen on buffers of the same size
    def parse(self, prgm: str) -> smpl.Function:
        tree = lark_parser.parse(prgm)
        # Strip the start rule and forward to recursive parsing function.
        return self._parse_body(tree)
