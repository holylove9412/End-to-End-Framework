import networkx as nx
import numpy as np
from swmm_api import read_inp_file

class SWMM_Converter:
    def __init__(self, inp_path, is_directed=False):
        self.inp_path = inp_path
        self.inp_network = read_inp_file(inp_path)
        self.lines = self._get_lines_from_textfile(inp_path)
        self.inp_dict = self._get_headers_from_inp()
        if is_directed:
            self.G = nx.MultiDiGraph()
        else:
            self.G = nx.Graph()

    def inp_to_G(self):
        self._add_nodes_to_G()
        self._add_edges_to_G()
        return self.G

    def _get_lines_from_textfile(self, path):
        with open(path, "r") as fh:
            lines = fh.readlines()
        return lines

    def _add_edges_to_G(self):
        # conduits = self._get_edge_attr_from_conduits_in_inp_lines()
        conduits = {(val.from_node,val.to_node,0):{'in_offset':val.offset_upstream,
                                                  'length':val.length,
                                                  'name_conduits':val.name,
                                                  'out_offset': val.offset_downstream,
                                                  'roughness':val.roughness,
                                                  'is_pump': 0.0,
                                                  'depth_on': np.nan,
                                                  'depth_off': np.nan,
                                                  } for key,val in
                                                  self.inp_network.CONDUITS.items()}
        pumps = {(val.from_node,val.to_node,int(val.name.split('.')[-1])): {'in_offset':np.nan,
                                                  'length':np.nan,
                                                  'name_conduits':val.name,
                                                  'out_offset': np.nan,
                                                  'roughness':np.nan,
                                              'is_pump':1,
                                              'depth_on':val.depth_on,
                                              'depth_off': val.depth_off,
                                              } for key,val in
                                                self.inp_network.PUMPS.items()}

        self.G.add_edges_from(conduits)
        self.G.add_edges_from(pumps)
        nx.set_edge_attributes(self.G, conduits)
        nx.set_edge_attributes(self.G, pumps)

        # x_sections = self._get_x_sections()
        x_sections = {key:{'conduit_shape':0.0,'height':val.height} for key,val in self.inp_network.XSECTIONS.items()} | {key:{'conduit_shape':np.nan,'height':np.nan} for key,val in self.inp_network.PUMPS.items()}

        phonebook = self._get_conduits_phonebook()
        self.G.graph["conduit_phonebook"] = phonebook
        new_x_sections = self._change_name_conduits_to_edge_tuple(phonebook, x_sections)
        nx.set_edge_attributes(self.G, new_x_sections)

    def _add_nodes_to_G(self):

        nodes_coordinates = {key:(val.x,val.y) for key,val in self.inp_network.COORDINATES.items()}
        nodes_elevation = {key:val.elevation for key,val in self.inp_network.JUNCTIONS.items()} | {key:val.elevation for key,val in self.inp_network.STORAGE.items()} | {key:val.elevation for key,val in self.inp_network.OUTFALLS.items()}

        nodes_max_depths = {key: val.depth_max for key, val in self.inp_network.JUNCTIONS.items()} | {key: val.depth_max
                                                                                                     for key, val in
                                                                                                     self.inp_network.STORAGE.items()} | {
                              key: 3.0 for key, val in self.inp_network.OUTFALLS.items()}
        nodes_is_outfall = {node_id: (1.0 if node_id in list(self.inp_network.OUTFALLS.keys()) else 0.0) for node_id in list(self.inp_network.COORDINATES.keys())}
        nodes_subcatchment = {val.outlet:key for key,val in self.inp_network.SUBCATCHMENTS.items()}
        # nodes_coordinates = self._get_nodes_coordinates()
        # nodes_elevation = self._get_nodes_elevation()
        # nodes_max_depths = self._get_nodes_max_depth()
        # nodes_is_outfall = self._get_nodes_is_outfall()
        # nodes_subcatchment = self._get_nodes_subcatchment()

        nodes_without_subcatchment = set(nodes_coordinates.keys()) - set(
            nodes_subcatchment.keys()
        )

        for node in nodes_without_subcatchment:
            nodes_subcatchment[node] = "None"

        nodes_names = list(nodes_coordinates.keys())
        nodes_names_dict = dict(zip(nodes_names, nodes_names))
        self.G.add_nodes_from(nodes_names)

        nx.set_node_attributes(self.G, nodes_coordinates, "pos")
        nx.set_node_attributes(self.G, nodes_elevation, "elevation")
        nx.set_node_attributes(self.G, nodes_max_depths, "max_depth")
        nx.set_node_attributes(self.G, nodes_is_outfall, "is_outfall")
        nx.set_node_attributes(self.G, nodes_names_dict, "name_nodes")
        nx.set_node_attributes(self.G, nodes_subcatchment, "subcatchment")

    def _get_headers_from_inp(self):
        inp_dict = dict()
        inp_dict = {
            line: number for (number, line) in enumerate(self.lines) if line[0] == "["
        }
        return inp_dict

    def _get_nodes_coordinates(self):
        index = self.inp_dict["[COORDINATES]\n"] + 3
        line = self.lines[index]
        pos = {}
        while line != "\n":
            name_node, x_coord, y_coord = line.split()
            pos[name_node] = (float(x_coord), float(y_coord))
            index += 1
            line = self.lines[index]
        return pos

    def _get_nodes_elevation(self):
        nodes_elevation = {}
        types_nodes = ["[JUNCTIONS]\n", "[OUTFALLS]\n", "[STORAGE]\n"]
        for type_of_node in types_nodes:
            try:
                elevations = self._get_elevation_from_type(type_of_node)
                nodes_elevation.update(elevations)
            except Exception as e:
                pass
                # print('The file does not have '+type_of_node)
        return nodes_elevation

    def _get_elevation_from_type(self, type_of_node):
        nodes_elevation = {}
        index = self.inp_dict[type_of_node] + 3
        line = self.lines[index]
        while line != "\n":
            if ";" not in line:
                name, elevation = line.split()[0], line.split()[1]
                nodes_elevation[name] = float(elevation)
            index += 1
            line = self.lines[index]
        return nodes_elevation

    def _get_nodes_max_depth(self):
        nodes_max_depth = {}
        types_nodes = ["[JUNCTIONS]\n", "[OUTFALLS]\n", "[STORAGE]\n"]
        for type_of_node in types_nodes:
            try:
                elevations = self._get_max_depth_from_type(type_of_node)
                nodes_max_depth.update(elevations)
            except Exception as e:
                pass
                # print('Something wrong with the nodes max depth'+ str(e))
        return nodes_max_depth

    def _get_max_depth_from_type(self, type_of_node):
        nodes_max_depth = {}

        if type_of_node == "[OUTFALLS]\n":
            index = self.inp_dict[type_of_node] + 3
            line = self.lines[index]
            while line != "\n":
                if ";" not in line:
                    name, max_depth = (
                        line.split()[0],
                        3.0,
                    )  # ! This number is hardcoded, it is 3 meters to give room for the outfall, but it should consider the setting (FREE, FIXED, NORMAL, TIDAL, etc.)
                    nodes_max_depth[name] = float(max_depth)
                index += 1
                line = self.lines[index]
        else:
            index = self.inp_dict[type_of_node] + 3
            line = self.lines[index]
            while line != "\n":
                if ";" not in line:
                    name, max_depth = (
                        line.split()[0],
                        line.split()[2],
                    )  # line.split()[2]
                    nodes_max_depth[name] = float(max_depth)
                index += 1
                line = self.lines[index]

        return nodes_max_depth

    def _get_nodes_is_outfall(self):
        nodes_is_outfall = {}
        types_nodes = ["[JUNCTIONS]\n", "[OUTFALLS]\n", "[STORAGE]\n"]
        for type_of_node in types_nodes:
            try:
                is_outfall = self._get_is_outfall_from_type(type_of_node)
                nodes_is_outfall.update(is_outfall)
            except Exception as e:
                pass
                # print('Something wrong with the nodes max depth'+ str(e))
        return nodes_is_outfall

    def _get_is_outfall_from_type(self, type_of_node):
        nodes_is_outfall = {}

        if type_of_node == "[OUTFALLS]\n":
            index = self.inp_dict[type_of_node] + 3
            line = self.lines[index]
            while line != "\n":
                if ";" not in line:
                    name, is_outfall = (
                        line.split()[0],
                        1.0,
                    )  # ! One means that it is an outfall
                    nodes_is_outfall[name] = float(is_outfall)
                index += 1
                line = self.lines[index]
        else:
            index = self.inp_dict[type_of_node] + 3
            line = self.lines[index]
            while line != "\n":
                if ";" not in line:
                    name, is_outfall = (
                        line.split()[0],
                        0.0,
                    )  # ! Zero means that it is not an outfall
                    nodes_is_outfall[name] = float(is_outfall)
                index += 1
                line = self.lines[index]

        return nodes_is_outfall

    def _get_nodes_subcatchment(self):
        nodes_subcatchment = {}
        index = self.inp_dict["[SUBCATCHMENTS]\n"] + 3
        line = self.lines[index]
        while line != "\n":
            if ";" not in line:
                l_split = line.split()
                nodes_subcatchment[l_split[2]] = l_split[0]
            index += 1
            line = self.lines[index]

        return nodes_subcatchment

    def _get_edge_attr_from_conduits_in_inp_lines(self):
        edge_attr_in_conduits = {}
        # Conduits
        index = self.inp_dict["[CONDUITS]\n"] + 3
        line = self.lines[index]
        while line != "\n":
            if ";" not in line:
                edge_attributes = {}

                l_split = line.split()
                source_node, destiny_node = l_split[1], l_split[2]

                edge_attributes["name_conduits"] = l_split[0]
                edge_attributes["length"] = float(l_split[3])
                edge_attributes["roughness"] = float(l_split[4])
                edge_attributes["in_offset"] = float(l_split[5])
                edge_attributes["out_offset"] = float(l_split[6])

                edge_attr_in_conduits[(source_node, destiny_node)] = edge_attributes

            index += 1
            line = self.lines[index]
        # Orifices
        try:
            index = self.inp_dict["[ORIFICES]\n"] + 3
            line = self.lines[index]
            while line != "\n":
                if ";" not in line:
                    edge_attributes = {}

                    l_split = line.split()
                    source_node, destiny_node = l_split[1], l_split[2]

                    edge_attributes["name_conduits"] = l_split[0]
                    edge_attributes["length"] = (
                        -999
                    )  # ! Dummy value to indicate it is an orifice
                    edge_attributes["roughness"] = (
                        -999
                    )  # ! Dummy value to indicate it is an orifice
                    edge_attributes["in_offset"] = (
                        -999
                    )  # ! Dummy value to indicate it is an orifice
                    edge_attributes["out_offset"] = (
                        -999
                    )  # ! Dummy value to indicate it is an orifice

                    edge_attr_in_conduits[(source_node, destiny_node)] = edge_attributes

                index += 1
                line = self.lines[index]
        except:
            pass

        return edge_attr_in_conduits

    def _get_x_sections(self):
        x_sections = {}
        index = self.inp_dict["[XSECTIONS]\n"] + 3
        line = self.lines[index]

        while line != "\n":
            if ";" not in line:
                x_sections_attributes = {}
                l_split = line.split()

                x_sections_attributes["conduit_shape"] = l_split[1]
                x_sections_attributes["geom_1"] = float(l_split[2])
                x_sections_attributes["geom_2"] = float(l_split[3])
                # It can continue, but I don't use the rest of the values

                x_sections[l_split[0]] = x_sections_attributes
            index += 1
            line = self.lines[index]
        return x_sections

    def _get_conduits_phonebook(self):
        name_dict = nx.get_edge_attributes(self.G, "name_conduits")
        name_tuple = {name_dict[k]: k for k in name_dict}
        return name_tuple

    def _change_name_conduits_to_edge_tuple(self, phonebook, edge_raw_data):
        renamed_edge_raw_data = {
            phonebook[k]: value for k, value in edge_raw_data.items()
        }
        return renamed_edge_raw_data

    def _get_subcatchments(self):
        subcathments = {}
        index = self.inp_dict["[SUBCATCHMENTS]\n"] + 3
        line = self.lines[index]

        while line != "\n":
            if ";" not in line:
                subcatchment_attributes = {}
                l_split = line.split()

                subcatchment_attributes["name_subcatchment"] = l_split[0]
                subcatchment_attributes["raingage"] = l_split[1]
                # The name of the outlet goes as key in the dictionary.
                subcatchment_attributes["area_subcatchment"] = l_split[3]
                # It can continue, but I don't use the rest of the values

                subcathments[l_split[2]] = subcatchment_attributes
            index += 1
            line = self.lines[index]
        return subcathments
