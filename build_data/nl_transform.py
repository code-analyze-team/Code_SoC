"""
a class that turn jimple representations into nl sentences
"""
# TODO: 1. empty types and values; 2. goto; 3. type ahead

import re
from build_data.preprocess import preprocessor
class nl_transformer:

    def __init__(self):
        self.info = 'jimple to natural language'
        self.label = None

    def label2tokens(self, label):
        # preprocess
        label = preprocessor.clean_label(label)
        nl = self.determine_and_solve(label)
        nl = nl.strip('\n')
        nl = nl.strip('\"')
        nl = nl.strip('\'')
        tokens = preprocessor.tokenize(nl)
        tokens = preprocessor.remove_stop_words(tokens)
        tokens = preprocessor.lemmatize(tokens)
        tokens = preprocessor.stemm(tokens)
        tokens = preprocessor.clean_nl_tokens(tokens)
        return tokens

    def determine_and_solve(self, label):
        self.label = preprocessor.clean_label(label)
        if label.find('JAssignStmt') != -1:
            nl = self.solve_assign()
        elif label.find('IdentityStmt') != -1:
            nl = self.solve_idntity()
        elif label.find('AbstractDefinitionStmt') != -1:
            nl = self.solve_abstract_def()
        elif label.find('JEnterMonitorStmt') != -1:
            nl = self.solve_enter_monitor()
        elif label.find('JExitMonitorStmt') != -1:
            nl = self.solve_exit_monitor()
        elif label.find('JReturn') != -1:
            nl = self.solve_return()
        elif label.find('JThrowStmt') != -1:
            nl = self.solve_throw()
        elif label.find('JLookupSwitchStmt') != -1:
            nl = self.solve_lookup_switch()
        elif label.find('JTableSwitchStmt') != -1:
            nl = self.solve_table_switch()
        elif label.find('JBreakpointStmt') != -1:
            nl = self.solve_break()
        elif label.find('JGotoStmt') != -1:
            nl = self.solve_goto()
        elif label.find('JIfStmt') != -1:
            nl = self.solve_if()
        elif label.find('JInvokeStmt') != -1:
            nl = self.solve_invoke()
        elif label.find('JNopStmt') != -1:
            nl = self.solve_nop()
        elif label.find('JRetStmt') != -1:
            nl = self.solve_ret()
        elif label.find('JRetVoidStmt') != -1:
            nl = self.solve_ret_void()
        elif label.find('PlaceholderStmt') != -1:
            nl = self.solve_placeholder()
        elif label.find('AbstractStmt') != -1:
            nl = self.solve_abstract()
        else:
            nl = self.solve_exception()
        self.nl = nl
        return nl

    def solve_assign(self):
        if self.label.find('JAssignStmt') != -1 and self.label.find('interfaceinvoke') != -1:
            nl = solve_assign_interfaceinvoke(self.label)
        elif self.label.find('JAssignStmt') != -1 and self.label.find('staticinvoke') != -1:
            nl = solve_assign_static_invoke(self.label)
        elif self.label.find('JAssignStmt') != -1 and self.label.find('virtualinvoke') != -1:
            nl = solve_virtual_invoke(self.label)
        elif self.label.find('JAssignStmt') != -1 and self.label.find('specialinvoke') != -1:
            nl = solve_assign_special_invoke(self.label)
        elif self.label.find('JAssignStmt') != -1 and self.label.find('dynamicinvoke') != -1:
            nl = solve_assign_dynamic_invoke(self.label)
        elif self.label.find('JAssignStmt') != -1 and \
            self.label.find('dynamicinvoke') == -1 and self.label.find('specialinvoke') == -1 \
            and self.label.find('virtualinvoke') == -1 and self.label.find('staticinvoke') == -1 and self.label.find('interfaceinvoke') == -1:
                nl = solve_assign_others(self.label)
        else:
            print('exception in assign')

        return nl

    def solve_invoke(self):
        if self.label.find('JInvokeStmt_interfaceinvoke') != -1:
            nl = solve_interface_invoke(self.label)
        elif self.label.find('JInvokeStmt_staticinvoke') != -1:
            nl = solve_static_invoke(self.label)
        elif self.label.find('JInvokeStmt_virtualinvoke') != -1:
            nl = solve_virtual_invoke(self.label)
        elif self.label.find('JInvokeStmt_dynamicinvoke') != -1:
            nl = solve_dynamic_invoke(self.label)
        elif self.label.find('JInvokeStmt_specialinvoke') != -1:
            nl = solve_special_invoke(self.label)
        else:
            nl = 'exception in invoke'
        return nl

    def solve_idntity(self):
        nl = solve_identities(self.label)
        return nl

    def solve_abstract_def(self):
        return 'we\'ve got some abstract_defs'

    def solve_enter_monitor(self):
        return 'we\'ve got some enter monitors'

    def solve_exit_monitor(self):
        return 'we\'ve got some exit monitors'

    def solve_return(self):
        nl = solve_returns(self.label)
        return nl

    def solve_throw(self):
        nl = solve_throws(self.label)
        return nl

    def solve_lookup_switch(self):
        return 'we\'ve got some lookup switches'

    def solve_table_switch(self):
        return 'we\'ve got some table switches'

    def solve_break(self):
        return 'we\'ve got some breaks'

    def solve_goto(self):
        nl = solve_gotos(self.label)
        return nl

    def solve_if(self):
        nl = solve_ifs(self.label)
        return nl

    def solve_nop(self):
        return 'got some nops'

    def solve_ret(self):
        return 'got some rets'

    def solve_ret_void(self):
        return 'got some ret voids'

    def solve_placeholder(self):
        return 'got some placeholders'

    def solve_abstract(self):
        return 'got some abstracts'

    def solve_exception(self):
        if len(self.label.split('.')) > 1:
            return 'a third party class or method'
        else:
            return 'an edge of type ' + self.label


def solve_interface_invoke(input):
    """
    example:
    "JInvokeStmt_interfaceinvoke $r3.<org.apache.ibatis.session.SqlSession: void select(java.lang.String,org.apache.ibatis.session.ResultHandler)>(r1, r2)"
    """
    pattern = re.compile(r'(.*)\s(.*)\.<(.*):\s(.*)\s(.*)\((.*)\)>\((.*)\)')
    # pattern = re.compile(r'(.*)_(.*)\s(.*)\.<(.*):\s(.*)\s(.*)\((.*)')
    res = re.search(pattern, input)
    assert res != None
    jtype = res.group(1)
    basic_type = jtype.split('_')[0]
    special_type = jtype.split('_')[1]
    caller = res.group(2)
    belong = res.group(3)
    ret_type = res.group(4)
    md_name = res.group(5)
    param_types = res.group(6).split(',')
    param_feeds= res.group(7).split(',')
    assert len(param_types) == len(param_feeds)
    nl = caller + ' invokes an interface method from interface ' + belong + '. The method has name as ' + md_name + ', has parameter types as '
    for param_type in param_types:
        nl += param_type + ', '
    nl += 'and are feed with parameter values as '
    for param_feed in param_feeds:
        nl += param_feed + ', '
    nl += 'Finally, the method returns as type ' + ret_type
    return nl

def solve_static_invoke(input):
    """
    example
    "JInvokeStmt_staticinvoke <org.junit.jupiter.api.Assertions: java.lang.Object fail()>()"
    """
    pattern = re.compile(r'(.*)\s<(.*):\s(.*)\s(.*)\((.*)\)>\((.*)\)')
    res = re.search(pattern, input)
    assert res != None
    jtype = res.group(1)
    basic_type = jtype.split('_')[0]
    special_type = jtype.split('_')[1]
    belong = res.group(2)
    ret_type = res.group(3)
    md_name = res.group(4)
    param_types = res.group(5).split(',')
    param_feeds = res.group(6).split(',')
    assert len(param_types) == len(param_feeds)
    nl = 'the method' + ' invokes an static method from class ' + belong + '. The method has name as ' + md_name + ', has parameter types as '
    for param_type in param_types:
        nl += param_type + ', '
    nl += 'and are feed with parameter values as '
    for param_feed in param_feeds:
        nl += param_feed + ', '
    nl += 'Finally, the method returns as type ' + ret_type
    return nl

def solve_virtual_invoke(input):
    """
        example:
        "JInvokeStmt_virtualinvoke r1.<org.springframework.context.support.GenericApplicationContext: void registerBeanDefinition(java.lang.String,org.springframework.beans.factory.config.BeanDefinition)>(\"sqlSessionTemplate\", $r4)"
        """
    pattern = re.compile(r'(.*)\s(.*)\.<(.*):\s(.*)\s(.*)\((.*)\)>\((.*)\)')
    # pattern = re.compile(r'(.*)_(.*)\s(.*)\.<(.*):\s(.*)\s(.*)\((.*)')
    res = re.search(pattern, input)
    assert res != None
    jtype = res.group(1)
    basic_type = jtype.split('_')[0]
    special_type = jtype.split('_')[1]
    caller = res.group(2)
    belong = res.group(3)
    ret_type = res.group(4)
    md_name = res.group(5)
    param_types = res.group(6).split(',')
    param_feeds = res.group(7).split(',')

    nl = caller + ' invokes an instance method from class ' + belong + '. The method has name as ' + md_name + ', has parameter types as '
    for param_type in param_types:
        nl += param_type + ', '
    nl += 'and are feed with parameter values as '
    for param_feed in param_feeds:
        nl += param_feed + ', '
    nl += 'Finally, the method returns as type ' + ret_type
    return nl

def solve_dynamic_invoke(input):
    """
    seems dynamic invoke dont exist without an assignment
    """

def solve_special_invoke(input):
    """
    example:
    "JInvokeStmt_specialinvoke $r3.<org.mybatis.spring.mapper.MapperFactoryBean: void <init>()>()"
    """
    pattern = re.compile(r'(.*)\s(.*)\.<(.*):\s(.*)\s(.*)\((.*)\)>\((.*)\)')
    # pattern = re.compile(r'(.*)_(.*)\s(.*)\.<(.*):\s(.*)\s(.*)\((.*)')
    res = re.search(pattern, input)
    assert res != None
    jtype = res.group(1)
    basic_type = jtype.split('_')[0]
    special_type = jtype.split('_')[1]
    caller = res.group(2)
    belong = res.group(3)
    ret_type = res.group(4)
    md_name = res.group(5)
    param_types = res.group(6).split(',')
    param_feeds = res.group(7).split(',')
    # assert len(param_types) == len(param_feeds)
    nl = caller + ' invokes an special method (init or private or from parent class) from interface ' + belong + '. The method has name as ' + md_name + ', has parameter types as '
    for param_type in param_types:
        nl += param_type + ', '
    nl += 'and are feed with parameter values as '
    for param_feed in param_feeds:
        nl += param_feed + ', '
    nl += 'Finally, the method returns as type ' + ret_type
    return nl

def solve_assign_interfaceinvoke(input):
    """
    example:
    "JAssignStmt_$r3 = interfaceinvoke $r2.<org.apache.ibatis.session.SqlSession: java.lang.Object selectOne(java.lang.String,java.lang.Object)>(\"org.mybatis.spring.sample.mapper.UserMapper.getUser\", r1)"
    :return:
    """
    pattern = re.compile(r'(.*)_(.*)\s=\s(.*)\s(.*)\.<(.*):\s(.*)\s(.*)\((.*)\)>\((.*)\)')
    res = re.search(pattern,input)
    assert res != None
    basic_type = res.group(1)
    assign_target = res.group(2)
    special_type = res.group(3)
    caller = res.group(4)
    belong = res.group(5)
    ret_type = res.group(6)
    md_name = res.group(7)
    param_types = res.group(8).split(',')
    param_feeds = res.group(9).split(',')

    nl = 'assign a value to variable ' + assign_target + '. '
    nl += caller + ' invokes an interface method from interface ' + belong + '. '
    nl += 'the method has name as ' + md_name + ', has parameter types as '
    for param_type in param_types:
        nl += param_type + ', '
    nl += 'and are feed with parameter values as '
    for param_feed in param_feeds:
        nl += param_feed + ', '
    nl += 'and returns a value of type ' + ret_type + '.'
    nl += 'Finally, the returned value of type ' + ret_type + ' is assigned to ' + assign_target + '.'
    return nl

def solve_assign_static_invoke(input):
    """
    "JAssignStmt_$r6 = staticinvoke <org.mybatis.spring.sample.config.SampleJobConfig: org.springframework.core.convert.converter.Converter createItemToParameterMapConverter(java.lang.String,java.time.LocalDateTime)>(\"batch_java_config_user\", $r5)"
    """
    pattern = re.compile(r'(.*)_(.*)\s=\s(.*)\s<(.*):\s(.*)\s(.*)\((.*)\)>\((.*)\)')
    res = re.search(pattern, input)
    assert res != None
    basic_type = res.group(1)
    assign_target = res.group(2)
    special_type = res.group(3)
    caller = ''
    belong = res.group(4)
    ret_type = res.group(5)
    md_name = res.group(6)
    param_types = res.group(7).split(',')
    param_feeds = res.group(8).split(',')

    nl = 'assign a value to variable ' + assign_target + '. '
    nl += caller + ' invokes an static method from class ' + belong + '. '
    nl += 'the method has name as ' + md_name + ', has parameter types as '
    for param_type in param_types:
        nl += param_type + ', '
    nl += 'and are feed with parameter values as '
    for param_feed in param_feeds:
        nl += param_feed + ', '
    nl += 'and returns a value of type ' + ret_type + '.'
    nl += 'Finally, the returned value of type ' + ret_type + ' is assigned to ' + assign_target + '.'
    return nl

def solve_assign_virtual_invoke(input):
    pattern = re.compile(r'(.*)_(.*)\s=\s(.*)\s(.*)\.<(.*):\s(.*)\s(.*)\((.*)\)>\((.*)\)')
    res = re.search(pattern, input)
    assert res != None
    basic_type = res.group(1)
    assign_target = res.group(2)
    special_type = res.group(3)
    caller = res.group(4)
    belong = res.group(5)
    ret_type = res.group(6)
    md_name = res.group(7)
    param_types = res.group(8).split(',')
    param_feeds = res.group(9).split(',')

    nl = 'assign a value to variable ' + assign_target + '. '
    nl += caller + ' invokes an virtual (instance) method from class ' + belong + '. '
    nl += 'the method has name as ' + md_name + ', has parameter types as '
    for param_type in param_types:
        nl += param_type + ', '
    nl += 'and are feed with parameter values as '
    for param_feed in param_feeds:
        nl += param_feed + ', '
    nl += 'and returns a value of type ' + ret_type + '.'
    nl += 'Finally, the returned value of type ' + ret_type + ' is assigned to ' + assign_target + '.'
    return nl

def solve_assign_special_invoke(input):
    pattern = re.compile(r'(.*)_(.*)\s=\s(.*)\s(.*)\.<(.*):\s(.*)\s(.*)\((.*)\)>\((.*)\)')
    res = re.search(pattern, input)
    assert res != None
    basic_type = res.group(1)
    assign_target = res.group(2)
    special_type = res.group(3)
    caller = res.group(4)
    belong = res.group(5)
    ret_type = res.group(6)
    md_name = res.group(7)
    param_types = res.group(8).split(',')
    param_feeds = res.group(9).split(',')

    nl = 'assign a value to variable ' + assign_target + '. '
    nl += caller + ' invokes an special (init, private, parent class) method from class ' + belong + '. '
    nl += 'the method has name as ' + md_name + ', has parameter types as '
    for param_type in param_types:
        nl += param_type + ', '
    nl += 'and are feed with parameter values as '
    for param_feed in param_feeds:
        nl += param_feed + ', '
    nl += 'and returns a value of type ' + ret_type + '.'
    nl += 'Finally, the returned value of type ' + ret_type + ' is assigned to ' + assign_target + '.'
    return nl

def solve_assign_dynamic_invoke(input):
    pattern = re.compile(r'(.*)_(.*)\s=\s(.*)\"\s<(.*),\shandle:\s<(.*):\s(.*)\s(.*)\((.*)\)>,\s(.*)\((.*)\)(.*)\)')
    res = re.search(pattern, input)
    assert res != None
    basic_type = res.group(1)
    assign_target = res.group(2)
    special = res.group(3)
    caller = ''
    special1 = special.split(' ')[0]
    special2 = special.split(' ')[1].replace('\"', '').replace('\\', '')
    lambda_stuff = res.group(4)
    belong = res.group(5)
    ret_type = res.group(6)
    md_name = res.group(7)
    param_types = res.group(8).split(',')
    wtf = res.group(9)
    feed_types = res.group(10).split(';')
    wtf2 = res.group(11)

    nl = 'assign a value to variable ' + assign_target + '. '
    nl += caller + ' invokes an dynamic method (lambda expression) from class' + belong + ' in mode ' + special2 + '. '
    nl += 'the method has name as ' + md_name + ', has parameter types as '
    for param_type in param_types:
        nl += param_type + ', '
    nl += 'and returns a value of type ' + ret_type + '.'
    nl += 'Finally, the returned value of type ' + ret_type + ' is assigned to ' + assign_target + '.'
    return nl

def solve_assign_others(input):
    left = input.split(' = ')[0]
    right = str(input.split(' = ')[1]).replace('\n', '')
    basic_type = left.split('_')[0]
    target = left.split('_')[1]
    target_ = re.match(r'(.*)\.<(.*):\s(.*)\s(.*)>', target)
    nl = None


    if re.match(r'(.*)\.<(.*):\s(.*)\s([^\(\)]*)>', right) != None:
        # object assign
        res_obj = re.match(r'(.*)\.<(.*):\s(.*)\s([^\(\)]*)>', right)
        assert res_obj != None
        right_caller = res_obj.group(1)
        right_belong = res_obj.group(2)
        right_type = res_obj.group(3)
        right_obj = res_obj.group(4)


        if target_ != None:
            assigned = target_.group(2) + '.' + target_.group(4)
            nl = 'assign an object to and object. '
            nl += right_caller + ' invokes an object from class ' + right_belong + ', '
            nl += 'the object has name as ' + right_obj + ' and is of type ' + right_type + '. '
            nl += 'the object was assigned to another object ' + assigned + '.'
        else:
            assigned = target
            nl = 'assign an object to and variable. '
            nl += right_caller + ' invokes an object from class ' + right_belong + ', '
            nl += 'the object has name as ' + right_obj + ' and is of type ' + right_type + '. '
            nl += 'the object was assigned to a variable ' + assigned + '.'


    elif re.match(r'\$*r[0-9]*', right) != None and len(right.split()) == 1:
        # variable assign
        res_var = re.match(r'\$*r[0-9]*', right)
        assert res_var != None

        if target_ != None:
            assigned = target_.group(2) + '.' + target_.group(4)
            nl = 'assign a variable ' + right + ' to an object ' + assigned + '.'
        else:
            assigned = target
            nl = 'assign a variable ' + right + 'to another variable ' + assigned + '.'

    elif re.match(r'(.*)\s([\+\-\*\/])\s(.*)', right) != None:
        # var op assign
        res_var_op = re.match(r'(.*)\s([\+\-\*\/])\s(.*)', right)
        assert res_var_op != None
        var1 = res_var_op.group(1)
        op = res_var_op.group(2)
        var2 = res_var_op.group(3)

        if target_ != None:
            assigned = target_.group(2) + '.' + target_.group(4)
            nl = 'assign the calculation result of ' + var1 + ' and ' + var2 + ' to an object ' + assigned + '.'
        else:
            assigned = target
            nl = 'assign the calculation result of ' + var1 + ' and ' + var2 + ' to an variable ' + assigned + '.'

    elif re.match(r'new\s(.*)', right) != None:
        # new obj assign
        res_newobj = re.match(r'new\s(.*)', right)
        assert res_newobj != None
        obj = res_newobj.group(1)
        if target_ != None:
            assigned = target_.group(2) + '.' + target_.group(4)
            nl = 'new an object ' + obj + ' and assign it to an object ' + assigned + '.'
        else:
            assigned = target
            nl = 'new an object ' + obj + ' and assign it to an variable ' + assigned + '.'
    elif re.match(r'newarray\s\((.*)\)\[(.*)\]', right) != None:
        # new array assign
        res_newarray = re.match(r'newarray\s\((.*)\)\[(.*)\]', right)
        assert res_newarray != None
        array = res_newarray.group(1)
        if target_ != None:
            assigned = target_.group(2) + '.' + target_.group(4)
            nl = 'new an array ' + array + ' and assign it to an object ' + assigned + '.'
        else:
            assigned = target
            nl = 'new an object ' + array + ' and assign it to an variable ' + assigned + '.'
    elif re.match(r'(.*)\sinstanceof\s(.*)', right) != None:
        # instanceof assign
        res_instance = re.match(r'(.*)\sinstanceof\s(.*)', right)
        assert res_instance != None
        v1 = res_instance.group(1)
        v2 = res_instance.group(2)
        if target_ != None:
            assigned = target_.group(2) + '.' + target_.group(4)
            nl = v1 + ' gets an instance of ' + v2 + ' and assign it to an object ' + assigned + '.'
        else:
            assigned = target
            nl = v1 + ' gets an instance of ' + v2 + ' and assign it to an variable ' + assigned + '.'
    elif re.match(r'\"(.*)\"', right) != None and re.match(r'\"([0-9]+)\"', right) == None:
        # string value assign
        res_value = re.match(r'\"(.*)\"', right)
        assert res_value != None
        value = res_value.group(1)
        if target_ != None:
            assigned = target_.group(2) + '.' + target_.group(4)
            nl = 'assign a value ' + value + '  to an object ' + assigned + '.'
        else:
            assigned = target
            nl = 'assign a value ' + value + ' to an variable ' + assigned + '.'
    elif re.match(r'\((.*)\)\s\$*r[0-9]+', right) != None:
        # type cast assign
        res_cast = re.match(r'\((.*)\)\s(\$*r[0-9]+)', right)
        assert res_cast != None
        cast = res_cast.group(1)
        casted = res_cast.group(2)
        if target_ != None:
            assigned = target_.group(2) + '.' + target_.group(4)
            nl = 'cast the type of variable ' + casted + ' to type ' + cast + ' and assign the variable to an object ' + assigned + '.'
        else:
            assigned = target
            nl = 'cast the type of variable ' + casted + ' to type ' + cast + ' and assign the variable to an variable ' + assigned + '.'
    elif re.match(r'lengthof\s(.*)', right) != None or re.match(r'\"*([0-9]+)\"*', right) != None:
        # num value assign
        if target_ != None:
            assigned = target_.group(2) + '.' + target_.group(4)
            nl = 'assign a number to an object ' + assigned + '.'
        else:
            assigned = target
            nl = 'assign a value to an variable ' + assigned + '.'
    else:
        nl = 'exception in assign_others'
    assert nl != None
    return nl

def solve_identities(input):
    pattern = re.compile(r'(.*)_(.*)\s\:\=\s\@(.*)')
    res = re.match(pattern, input)
    assert res != None
    basic_type = res.group(1)
    target = res.group(2)
    value = res.group(3)
    if len(value.split(': ')) > 1:
        nl = 'define variable ' + target + ' as ' + value.split(': ')[1] + '.'
    else:
        nl = 'define variable ' + target + ' as ' + value + '.'
    return nl

def solve_returns(input):
    if input.find('JReturnVoidStmt') != -1:
        nl = 'returns a void value.'
    else:
        pattern = re.compile(r'JReturnStmt_return\s(.*)')
        value = re.match(pattern, input).group(1)
        assert value != None
        value = value.replace('\"', '').replace('\'', '')
        nl = 'returns a value as : ' + value + '.'
    return nl

def solve_throws(input):
    pattern = re.compile(r'JThrowStmt_throw\s(.*)')
    res = re.match(pattern, input)
    assert res != None
    value = res.group(1)
    nl = 'throws a value as : ' + value + '.'
    return nl

def solve_gotos(input):
    return 'goto'

def solve_ifs(input):
    pattern = re.compile(r'JIfStmt_if\s(.*)\sgoto\s(.*)')
    res= re.match(pattern, input)
    assert res != None
    condition = res.group(1)
    move = res.group(2)
    nl = 'condition: ' + condition + '.'
    return nl






