class Scanner:
    current = 0

    def scan(self, line):
        data = []

        def isValid(pos):
            return pos < len(line)

        def isDigit(pos):
            return line[pos].isdigit()

        def isAlpha():
            return line[self.current].isalpha()

        def number():
            num = 0
            while isValid(self.current) and isDigit(self.current):
                num = num * 10 + int(line[self.current])
                self.current += 1
            data.append(("INTEGER", num))

        def identifier():
            id = ""
            while isValid(self.current) and isAlpha():
                id += line[self.current]
                self.current += 1

            if id == "and":
                data.append(("AND", None))
            elif id == "or":
                data.append(("OR", None))
            elif id == "not":
                data.append(("NOT", None))
            elif id == "True":
                data.append((True, None))
            elif id == "False":
                data.append((False, None))
            else:
                data.append(("IDENTIFIER", id))

        while self.current < len(line):
            if line[self.current] == '+':
                data.append(("PLUS", None))
                self.current += 1
            elif line[self.current] == '-':
                data.append(("MINUS", None))
                self.current += 1
            elif line[self.current] == '*':
                data.append(("MULTIPLY", None))
                self.current += 1
            elif line[self.current] == '/':
                data.append(("DIVIDE", None))
                self.current += 1
            elif line[self.current] == '>':
                if isValid(self.current + 1) and line[self.current + 1] == '=':
                    data.append(("GREATER_EQUAL", None))
                    self.current += 2
                else:
                    data.append(("GT", None))
                    self.current += 1
            elif line[self.current] == '<':
                if isValid(self.current + 1) and line[self.current + 1] == '=':
                    data.append(("LESS_EQUAL", None))
                    self.current += 2
                else:
                    data.append(("LT", None))
                    self.current += 1
            elif line[self.current] == '=':
                if isValid(self.current + 1) and line[self.current + 1] == '=':
                    data.append(("EQUAL_EQUAL", None))
                    self.current += 2
                else:
                    data.append(("EQUAL", None))
                    self.current += 1
            elif line[self.current] == '!':
                if isValid(self.current+1) and line[self.current+1] == '=':
                    data.append(("NOT_EQUAL", None))
                    self.current += 2

            elif isDigit(self.current):
                number()

            elif isAlpha():
                identifier()

            elif line[self.current] == ' ':
                self.current += 1

            else:
                raise Exception("Invalid character")
        self.current = 0
        return data


class Parser:
    def __init__(self):
        self.data = []

    def parse(self, tokens):
        current = 0

        def isValid(pos):
            return pos < len(tokens)

        def isConstant(token):
            return token[0] == "INTEGER"

        def isVariable(token):
            return token[0] == "IDENTIFIER"

        def isBinaryExpression():
            if isValid(current+2) and isValid(current+3) and isValid(current+4):
                return True
            return False

        def findTuple(x):
            for v in self.data:
                if type(v) == tuple:
                    if v[0] == x[1]:
                        return v[1]
            raise Exception("Variable not found")

        def operateBinary(op, ta, tb):
            t1 = 0
            t2 = 0
            if isVariable(ta):
                t1loc = findTuple(ta)
                t1 = self.data[t1loc]
            else:
                t1 = ta[1]
            if isVariable(tb):
                t2loc = findTuple(tb)
                t2 = self.data[t2loc]
            else:
                t2 = tb[1]

            if op == "PLUS":
                return t1+t2
            elif op == "MINUS":
                return t1-t2
            elif op == "MULTIPLY":
                return t1*t2
            elif op == "DIVIDE":
                return t1//t2
            elif op == "GT":
                return t1 > t2
            elif op == "LT":
                return t1 < t2
            elif op == "GREATER_EQUAL":
                return t1 >= t2
            elif op == "LESS_EQUAL":
                return t1 <= t2
            elif op == "EQUAL_EQUAL":
                return t1 == t2
            elif op == "NOT_EQUAL":
                return t1 != t2
            elif op == "AND":
                return t1 and t2
            elif op == "OR":
                return t1 or t2

        def isUnaryExpression():
            if isValid(current+2) and isValid(current+3):
                return True
            return False

        def operateUnary(op, ta):
            t1 = 0
            if isVariable(ta):
                t1loc = findTuple(ta)
                t1 = t1loc[1]
            else:
                t1 = ta[1]

            if op == "MINUS":
                return -t1
            elif op == "NOT":
                return not t1

        def insert(i):
            if type(i) == int:
                if i in self.data:
                    return self.data.index(i)
                self.data.append(i)
                return len(self.data)-1
            elif isConstant(i):
                if i[1] in self.data:
                    return self.data.index(i[1])
                self.data.append(i[1])
                return len(self.data)-1
            else:
                tuploc = findTuple(i)
                return insert(self.data[tuploc])

        def replace(x, j):
            found = False
            for (ind, d) in enumerate(self.data):
                if type(d) == tuple:
                    if d[0] == x:
                        self.data[ind] = (x, j)
                        found = True
            if not found:
                self.data.append((x, j))

        while current < len(tokens):
            token = tokens[current]
            if isVariable(token):
                if isValid(current+1) and tokens[current+1][0] == 'EQUAL':
                    if isBinaryExpression():
                        insert(tokens[current+2])
                        insert(tokens[current+4])
                        res = operateBinary(
                            tokens[current+3][0], tokens[current+2], tokens[current+4])
                        pos = insert(res)
                        replace(token[1], pos)
                        current += 5
                    elif isUnaryExpression():
                        insert(tokens[current+1])
                        res = operateUnary(
                            tokens[current+1][1], tokens[current+1])
                        pos = insert(res)
                        replace(token[1], pos)
                        current += 4
                    elif isConstant(tokens[current+2]):
                        pos = insert(tokens[current+2])
                        replace(token[1], pos)
                        current += 3
                    elif isVariable(tokens[current+2]):
                        pos = findTuple(tokens[current+2])
                        replace(token[1], pos)
                        current += 3
                    else:
                        raise Exception("Invalid expression")


if __name__ == "__main__":
    line1 = "q = 6"
    line2 = "p = q+5"
    line3 = "r = p"

    scanner = Scanner()
    t1 = scanner.scan(line1)
    print("t1 = ", t1)
    t2 = scanner.scan(line2)
    print("t2 = ", t2)
    t3 = scanner.scan(line3)
    print("t3 = ", t3)

    parser = Parser()
    d1 = parser.parse(t1)
    print("d1 = ", parser.data)
    d2 = parser.parse(t2)
    print("d2 = ", parser.data)
    d3 = parser.parse(t3)
    print("d3 = ", parser.data)
