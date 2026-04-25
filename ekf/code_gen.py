import re

from sympy import Integer
from sympy.codegen.ast import float32, real
from sympy.printing.c import C99CodePrinter
from sympy.printing.precedence import precedence


class EkfC99CodePrinter(C99CodePrinter):
    def _print_Pow(self, expr):
        exp = expr.exp
        if isinstance(exp, Integer) and 2 <= int(exp) <= 4:
            base = self.parenthesize(expr.base, precedence(expr))
            return "*".join([base] * int(exp))
        return super()._print_Pow(expr)


class CodeGenerator:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file = open(self.file_name, "w")
        self.printer = EkfC99CodePrinter({"type_aliases": {real: float32}})

    def print_string(self, string):
        self.file.write("// " + string + "\n")

    def get_ccode(self, expression):
        return self.printer.doprint(expression)

    def write_subexpressions(self, subexpressions):
        write_string = ""
        for item in subexpressions:
            write_string = (
                write_string
                + "const float "
                + str(item[0])
                + " = "
                + self.get_ccode(item[1])
                + ";\n"
            )

        write_string = write_string + "\n\n"
        self.file.write(write_string)

    def write_matrix(self, matrix, variable_name, is_symmetric=False):
        write_string = ""

        if matrix.shape[0] * matrix.shape[1] == 1:
            write_string = (
                write_string + variable_name + " = " + self.get_ccode(matrix[0]) + ";\n"
            )
        elif matrix.shape[0] == 1 or matrix.shape[1] == 1:
            for i in range(0, len(matrix)):
                write_string = (
                    write_string
                    + variable_name
                    + "["
                    + str(i)
                    + "] = "
                    + self.get_ccode(matrix[i])
                    + ";\n"
                )
        else:
            for j in range(0, matrix.shape[1]):
                for i in range(0, matrix.shape[0]):
                    if j >= i or not is_symmetric:
                        write_string = (
                            write_string
                            + variable_name
                            + "["
                            + str(i)
                            + "]["
                            + str(j)
                            + "] = "
                            + self.get_ccode(matrix[i, j])
                            + ";\n"
                        )
        write_string = write_string + "\n\n"
        self.file.write(write_string)

    def close(self):
        self.file.close()


class RustCodeGenerator(CodeGenerator):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.file.write("{\n")

    def get_ccode(self, expression):
        code = super().get_ccode(expression)
        code = re.sub(r"(?<=\d)F\b", "_f32", code)
        code = re.sub(r"(?<![\[\]\w.])(\d+)(?![\[\]\w.])", r"\1.0_f32", code)
        return code

    def write_subexpressions(self, subexpressions):
        write_string = ""
        for item in subexpressions:
            write_string += (
                "let "
                + str(item[0])
                + ": f32 = "
                + self.get_ccode(item[1])
                + ";\n"
            )

        write_string = write_string + "\n\n"
        self.file.write(write_string)

    def close(self):
        self.file.write("}\n")
        super().close()
