from build_data.dot_parse import dot_parser
import networkx as nx

path1 = '/home/qwe/zfy_lab/project_dots/spring-master/AST_CFG_PDGdotInfo/'
path2 = '/home/qwe/projects/1.Spring core/AST_CFG_PDGdotInfo/'
path3 = '/home/qwe/projects/43.Spring Integration/AST_CFG_PDGdotInfo/'
path4 = '/home/qwe/projects/46.Spring batch/AST_CFG_PDGdotInfo/'

path = path1 + '#' + path2 + '#' + path3 + '#' + path4

parser = dot_parser(path)
print(parser.path)
parser.read_dot_files()
print(len(parser.cfg_dot_files))
# parser.map_method_dot()
# parser.map_dot_method()
# parser.get_relabelled_graphs()
#
# graphs = parser.relabelled_graph_list