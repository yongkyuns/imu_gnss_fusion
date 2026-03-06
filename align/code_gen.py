from sympy import Integer
import re
from sympy.codegen.ast import float32, real
from sympy.printing.c import C99CodePrinter
from sympy.printing.precedence import precedence
from sympy.printing.rust import RustCodePrinter


class MisalignmentC99CodePrinter(C99CodePrinter):
    def _print_Pow(self, expr):
        exp = expr.exp
        if isinstance(exp, Integer) and 2 <= int(exp) <= 4:
            base = self.parenthesize(expr.base, precedence(expr))
            return "*".join([base] * int(exp))
        return super()._print_Pow(expr)


class MisalignmentRustCodePrinter(RustCodePrinter):
    def _print_Pow(self, expr):
        exp = expr.exp
        if isinstance(exp, Integer) and 2 <= int(exp) <= 4:
            base = self.parenthesize(expr.base, precedence(expr))
            return "*".join([base] * int(exp))
        return super()._print_Pow(expr)


class CodeGenerator:
    def __init__(self, file_name, language):
        self.file_name = file_name
        self.file = open(self.file_name, "w")
        self.language = language
        if language == "c":
            self.printer = MisalignmentC99CodePrinter({"type_aliases": {real: float32}})
        elif language == "rust":
            self.printer = MisalignmentRustCodePrinter()
        else:
            raise ValueError(f"unsupported language: {language}")

    def print_string(self, string):
        if self.language == "c":
            self.file.write("// " + string + "\n")
        else:
            self.file.write("// " + string + "\n")

    def get_code(self, expression):
        code = self.printer.doprint(expression)
        if self.language == "rust":
            # Keep generated snippets in f32 domain for embedding in align.rs.
            code = code.replace("_f64", "_f32")
            code = re.sub(r"(?<![\w.])(\d+)(?![\w.])", r"\1.0", code)
        return code

    def write_subexpressions(self, subexpressions):
        lines = []
        for item in subexpressions:
            if self.language == "c":
                lines.append(f"const float {item[0]} = {self.get_code(item[1])};")
            else:
                lines.append(f"let {item[0]}: f32 = {self.get_code(item[1])};")
        self.file.write("\n".join(lines))
        self.file.write("\n\n")

    def write_vector(self, vector, variable_name):
        lines = []
        for i in range(len(vector)):
            lines.append(
                f"{variable_name}[{i}] = {self.get_code(vector[i])};"
            )
        self.file.write("\n".join(lines))
        self.file.write("\n\n")

    def write_matrix(self, matrix, variable_name):
        lines = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                lines.append(
                    f"{variable_name}[{i}][{j}] = {self.get_code(matrix[i, j])};"
                )
        self.file.write("\n".join(lines))
        self.file.write("\n\n")

    def close(self):
        self.file.close()
