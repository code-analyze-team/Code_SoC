from build_data.dot_parse import dot_parser
import networkx as nx

path = '/home/qwe/zfy_lab/project_dots/spring-master/AST_CFG_PDGdotInfo/'

parser = dot_parser(path)
parser.read_dot_files()
parser.map_method_dot()
parser.map_dot_method()
parser.get_relabelled_graphs()

graphs = parser.relabelled_graph_list