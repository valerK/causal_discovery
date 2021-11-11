from itertools import combinations
import numpy as np


class FastCausalInference:

    def __init__(self, data, conditional_independence_test):

        self.data = data
        sample_counts = [len(data[key]) for key in data]
        for sample_count in sample_counts:
            if sample_count != sample_counts[0]:
                raise ValueError('All Variables Should Contain the Same Number of Samples!')
        self.sample_count = sample_counts[0]
        self.conditional_independence_test = conditional_independence_test

    def initialize_edges(self, nodes):
        edges = set()
        for node_from in nodes:
            for node_to in nodes:
                if node_from != node_to:
                    edges.add((node_from, node_to))
        return edges

    def calculate_adjacencies(self, edges):
        adjacencies = {}
        for edge in edges:
            try:
                adjacencies[edge[0]].add(edge[1])
            except KeyError:
                adjacencies[edge[0]] = {edge[1]}
        return adjacencies

    def select_edge_with_l(self, edges, edges_checked, adjacencies, l):
        selected_edge = None
        for edge in edges:
            if len(adjacencies[edge[0]].difference({edge[1]})) >= l and edge not in edges_checked:
                selected_edge = edge
                break
        return selected_edge

    def create_edges_dict(self, edges, reorient_tails=False):
        edges_dict = {}
        for edge in edges:
            if edge[0] not in edges_dict:
                edges_dict[edge[0]] = {}
            if edge[1] not in edges_dict[edge[0]]:
                if reorient_tails:
                    edges_dict[edge[0]][edge[1]] = (edge[0], edge[1], 'o', 'o')
                else:
                    edges_dict[edge[0]][edge[1]] = edge
        return edges_dict

    def create_graph_dict(self, edges):
        graph_dict = {}
        for edge in edges:
            try:
                graph_dict[edge[0]].add(edge[1])
            except KeyError:
                graph_dict[edge[0]] = set()
                graph_dict[edge[0]].add(edge[1])
            try:
                graph_dict[edge[1]].add(edge[0])
            except KeyError:
                graph_dict[edge[1]] = set()
                graph_dict[edge[1]].add(edge[0])
        return graph_dict

    def orient_unshield_triples(self, edges, separation_set):

        graph_dict = self.create_graph_dict(edges)
        edges_dict = self.create_edges_dict(edges, True)

        unshielded_triples = set()
        for node_left in graph_dict:
            for node_middle in graph_dict[node_left]:
                for node_right in graph_dict[node_middle]:
                    if node_left == node_right:
                        continue
                    if node_left not in graph_dict[node_right]:
                        unshielded_triples.add((node_left, node_middle, node_right))

        for unshielded_triple in unshielded_triples:
            try:
                separations = separation_set[(unshielded_triple[0], unshielded_triple[2])]
            except KeyError:
                separations = []
            if (unshielded_triple[1],) not in separations:
                current_edge = edges_dict[unshielded_triple[0]][unshielded_triple[1]]
                edges_dict[unshielded_triple[0]][unshielded_triple[1]] = \
                    (current_edge[0], current_edge[1], current_edge[2], '>')
                if current_edge[2] == 'o':
                    arrowhead = 'o'
                else:
                    arrowhead = '>'
                edges_dict[unshielded_triple[1]][unshielded_triple[0]] = \
                    (current_edge[1], current_edge[0], '<', arrowhead)
            else:
                pass

        return [edges_dict[node_from][node_to] for node_from in edges_dict for node_to in edges_dict[node_from]]

    def edge_selection_based_on_adjacencies(self, data, edges):
        separation_set = {}
        l = 0
        while True:
            l += 1

            edges_checked = set()
            while True:
                adjacencies = self.calculate_adjacencies(edges)

                selected_edge = self.select_edge_with_l(edges, edges_checked, adjacencies, l)
                if selected_edge is None:
                    break

                edges_checked.add(selected_edge)
                for conditions in combinations(adjacencies[selected_edge[0]].difference({selected_edge[1]}), r=l):

                    x = np.array(data[selected_edge[0]]).reshape(-1, 1)
                    y = np.array(data[selected_edge[1]]).reshape(-1, 1)
                    z = np.concatenate([np.array(data[condition]).reshape(-1, 1) for condition in conditions], axis=1)

                    is_independent = self.conditional_independence_test(x, y, z)

                    if is_independent:
                        edges.remove(selected_edge)
                        edges.remove((selected_edge[1], selected_edge[0]))
                        separation_set[selected_edge] = conditions
                        break

            adjacencies = self.calculate_adjacencies(edges)
            check_adjacencies = True
            for edge in edges:
                if len(adjacencies[edge[0]].difference({edge[1]})) >= l:
                    check_adjacencies = False
                    break
            if check_adjacencies:
                break

        return edges, separation_set

    def check_path_eligibility(self, path, edges_dict):
        for i in range(len(path) - 2):
            sub_path = path[i:i + 3]
            is_collider = edges_dict[sub_path[0]][sub_path[1]][3] == '>' and edges_dict[sub_path[1]][sub_path[2]][
                2] == '<'
            is_triangle = \
                sub_path[1] in edges_dict[sub_path[0]] and \
                sub_path[2] in edges_dict[sub_path[1]] and \
                sub_path[0] in edges_dict[sub_path[2]]

            is_eligible = is_collider or is_triangle
            if not is_eligible:
                return False
        return True

    def get_possible_d_sep(self, edges_dict):
        nodes = list(edges_dict.keys())
        possible_d_sep = {}
        for node in nodes:
            for node_to in nodes:
                if node == node_to:
                    continue

                paths = [[node]]
                while True:
                    found_paths = []
                    paths_next = []
                    for path in paths:
                        neighbours = list(edges_dict[path[-1]].keys())
                        for neighbour in neighbours:
                            if neighbour == node_to:
                                found_paths.append(path + [neighbour])
                            elif neighbour not in path:
                                paths_next.append(path + [neighbour])
                            else:
                                pass
                    paths = paths_next
                    if len(paths) == 0:
                        break

                    variable_checked = False
                    for path in found_paths:
                        if len(path) >= 3:
                            if self.check_path_eligibility(path, edges_dict):
                                try:
                                    possible_d_sep[node].add(node_to)
                                except KeyError:
                                    possible_d_sep[node] = {node_to}
                                variable_checked = True
                                break

                    if variable_checked:
                        break

            if node not in possible_d_sep:
                possible_d_sep[node] = set()

        return possible_d_sep

    def edge_selection_based_on_possible_d_sep(self, data, edges, separation_set):

        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])

        edges_dict = self.create_edges_dict(edges, False)

        for node in nodes:
            possible_d_sep = self.get_possible_d_sep(edges_dict)

            for neighbour in list(edges_dict[node].keys()):

                for l in range(1, len(possible_d_sep[node]) + 1):

                    selected_combinations = \
                        [elem for elem in combinations(possible_d_sep[node].difference({neighbour}), r=l)]

                    if len(selected_combinations) == 0:
                        break

                    is_edge_deleted = False
                    for conditions in selected_combinations:

                        x = np.array(data[node]).reshape(-1, 1)
                        y = np.array(data[neighbour]).reshape(-1, 1)
                        z = np.concatenate(
                            [np.array(data[condition]).reshape(-1, 1) for condition in conditions], axis=1
                        )

                        is_independent = self.conditional_independence_test(x, y, z)

                        if is_independent:
                            del edges_dict[node][neighbour]
                            del edges_dict[neighbour][node]
                            separation_set[(node, neighbour)] = conditions
                            is_edge_deleted = True
                            break

                    if is_edge_deleted:
                        break

        return [edges_dict[node_from][node_to] for node_from in edges_dict for node_to in edges_dict[node_from]], \
               separation_set

    def infer_separation_set_for_edges(self, data, edges):
        nodes = set(data.keys())
        edges_to_check = {(edge[0], edge[1]) for edge in edges}

        missing_edges = set()
        for node_from in nodes:
            for node_to in nodes:
                if (node_from, node_to) not in edges_to_check:
                    missing_edges.add((node_from, node_to))

        separation_set = {}
        edges_dict = self.create_edges_dict(edges, False)
        for edge in missing_edges:
            separation_set[edge] = \
                {node_to for node_to in edges_dict[edge[0]]}.union({node_to for node_to in edges_dict[edge[1]]})

        return separation_set

    def infer_skeleton(self):

        edges = self.initialize_edges(list(self.data.keys()))
        edges, separation_set = self.edge_selection_based_on_adjacencies(self.data, edges)
        edges = self.orient_unshield_triples(edges, separation_set)
        edges, separation_set = self.edge_selection_based_on_possible_d_sep(self.data, edges, separation_set)
        edges = self.orient_unshield_triples(edges, separation_set)

        return edges

    def bootstrap_infer_skeleton(self, bootstrap_samples=100, bootstrap_sample_ratio=1.0, bootstrap_edge_threshold=0.95):

        edge_counts = {}
        for i in range(bootstrap_samples):
            print('Inferring bootstrap sample no.', i)
            sample_indices = np.random.choice(self.sample_count, size=int(self.sample_count * bootstrap_sample_ratio))

            data_sampled = {
                item: [self.data[item][sample_idx] for sample_idx in sample_indices]
                for item in self.data
            }

            edges = self.initialize_edges(list(data_sampled.keys()))
            edges, separation_set = self.edge_selection_based_on_adjacencies(data_sampled, edges)
            edges = self.orient_unshield_triples(edges, separation_set)

            edges, separation_set = self.edge_selection_based_on_possible_d_sep(data_sampled, edges, separation_set)

            for edge in edges:
                try:
                    edge_counts[(edge[0], edge[1], 'o', 'o')] += 1.0
                except KeyError:
                    edge_counts[(edge[0], edge[1], 'o', 'o')] = 1.0

        edge_counts = {key: edge_counts[key] / bootstrap_samples for key in edge_counts}
        edges = [key for key in edge_counts if edge_counts[key] >= bootstrap_edge_threshold]

        separation_set = self.infer_separation_set_for_edges(self.data, edges)
        edges = self.orient_unshield_triples(edges, separation_set)

        return edges
