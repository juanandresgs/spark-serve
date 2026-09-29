import json
import unittest
from tool_adapter import adapt_calls, strict_json, install

class AdapterTests(unittest.TestCase):
    def convert(self, value, schema):
        tools=[{'type':'function','function':{'name':'probe','parameters':{'type':'object','properties':{'value':schema}}}}]
        calls=[{'id':'fixture','type':'function','function':{'name':'probe','arguments':json.dumps({'value':value})}}]
        result=adapt_calls(calls,tools)
        self.assertEqual(json.loads(calls[0]['function']['arguments'])['value'],value)
        return json.loads(result[0]['function']['arguments'])['value']

    def test_parameter_matrix(self):
        schemas=[{'type':t} for t in ['string','integer','number','boolean','null','array','object']]
        values=['7','-8','1.25','true','false','null','[2,5]','{"a":1}','007','NaN','1e999',' three ','','π','{"a":1,"a":2}','[true,null]']
        for schema in schemas:
            for value in values:
                with self.subTest(schema=schema,value=value):
                    actual=self.convert(value,schema)
                    expected=value
                    if schema['type']!='string':
                        try:
                            parsed=strict_json(value)
                            types={'integer':type(parsed) is int,'number':type(parsed) in (int,float),'boolean':type(parsed) is bool,'null':parsed is None,'array':type(parsed) is list,'object':type(parsed) is dict}
                            if types.get(schema['type']):expected=parsed
                        except ValueError:pass
                    self.assertEqual(actual,expected)
                    self.assertEqual(type(actual),type(expected))

    def test_constraints_and_ambiguous_union(self):
        for schema in [{'type':['string','integer']},{}, {'anyOf':[{'type':'string'},{'type':'integer'}]}]:
            self.assertEqual(self.convert('7',schema),'7')
        self.assertEqual(self.convert('7',{'type':'integer','minimum':8}),'7')
        self.assertEqual(self.convert('[1,"2"]',{'type':'array','items':{'type':'integer'}}),'[1,"2"]')
        self.assertEqual(self.convert('{\\"a\\":1}',{'type':'object','properties':{'a':{'type':'integer'}}}),{'a':1})
        self.assertEqual(self.convert('  007\nπ\\path',{'type':'string'}),'  007\nπ\\path')

    def test_xml_boolean_spellings(self):
        for word,expected in [('True',True),('False',False)]:
            self.assertIs(self.convert(word,{'type':'boolean'}),expected)
            self.assertEqual(self.convert(word,{'type':'string'}),word)
            self.assertEqual(self.convert(word,{'type':'integer'}),word)
        self.assertEqual(self.convert('truthy',{'type':'boolean'}),'truthy')

    def test_external_schema_is_not_fetched(self):
        self.assertEqual(self.convert('7',{'$ref':'https://invalid.example/schema.json'}),'7')

    def test_installed_parser_boundary(self):
        install()
        from tensorfold.cuda.server import parse_tool_calls
        tools=[{'type':'function','function':{'name':'probe','parameters':{'type':'object','properties':{'n':{'type':'integer'},'s':{'type':'string'}}}}}]
        text='<tool_call><function=probe><parameter=n>7</parameter><parameter=s>007</parameter></function></tool_call>'
        for limit in [None,1]:
            content,calls=parse_tool_calls(text,tools,max_calls=limit)
            self.assertEqual(content,'')
            self.assertEqual(json.loads(calls[0]['function']['arguments']),{'n':7,'s':'007'})
        for text in ['plain response','<tool_call><function=unoffered></function></tool_call>','{"n":7}']:
            _,calls=parse_tool_calls(text,tools)
            self.assertIsNone(calls)
        _,calls=parse_tool_calls('<tool_call>{"name":"probe","arguments":{"n":9,"s":"007"}}</tool_call>',tools)
        self.assertEqual(json.loads(calls[0]['function']['arguments']),{'n':9,'s':'007'})

if __name__=='__main__':unittest.main()
