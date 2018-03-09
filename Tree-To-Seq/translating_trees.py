def tree_to_list(self, ast):

def list_to_tree(self, ls):

def translate_from_for(self, ls):
    if ls[0] == '<SEQ>':
        t1 = self.translate_from_for(ls[1])
        t2 = self.translate_from_for(ls[2])
        if t1[0] == '<LET>' and t1[-1] == '<UNIT>':
            t1[-1] = t2
            return t1
        else:
            return ['<LET>', 'blank', t1, t2]
    elif ls[0] == '<IF>':
        cmp = ls[1]
        t1 = self.translate_from_for(ls[2])
        t2 = self.translate_from_for(ls[3])
        return ['<IF>', cmp, t1, t2]
    elif ls[0] == '<FOR>':
        var = ls[1]
        init = self.translate_from_for(ls[2])
        cmp = self.translate_from_for(ls[3])
        inc = self.translate_from_for(ls[4])
        body = self.translate_from_for(ls[5])
        tb = ['<LET>', 'blank', body, ['<APP>', 'func', inc]]
        funcbody = ['<IF>', cmp, tb, '<UNIT>']
        translate = ['<LETREC>', 'func', var, funcbody, ['<APP>', 'func', init]]
        return translate
    elif ls[0] == '<ASSIGN>':
        return ['<LET>', ls[1], ls[2], '<UNIT>']
    else:
        return ls