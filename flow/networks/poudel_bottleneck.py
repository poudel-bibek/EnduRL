

# import Flow's base network class
from flow.networks import Network
import numpy as np
# Additional params are defined here
# They are accessed as net_params.additional_params["param"]
# Make sure to copy and pass the same net params when this network is used
ADDITIONAL_NET_PARAMS = {
    "speed_limit": 20,
}

class PoudelBottleneckNetwork(Network):
    """
    Nodes, edges, routes

    """
    
    def specify_nodes(self, net_params):
        """
        Positions of select few point in the network
        id: name,
        x, y: coordinates 
        Nodes dont need to consider lanes
        """
        nodes = [
            # Left: Start
            {"id": "left0", "x": 0, "y": 0},

            # Middle: 
            {"id": "middle0", "x": 250, "y": 0},
            {"id": "middle1", "x": 750, "y": 0},
            {"id": "middle2", "x": 1250, "y": 0},

            # Right: End
            {"id": "right0", "x": 1500, "y": 0},
        ]

        return nodes
    
    def specify_edges(self, net_params):
        """
        Nodes are connected by edges
        id: name,
        from: start node,
        to: end node,
        length: length of the edge,
        numLanes: number of lanes,
        speed: speed limit,
        other sumo related attributes: https://sumo.dlr.de/docs/Networks/PlainXML.html#Edge_Descriptions
        One useful attribute is "shape" which is a list of coordinates (a series of sub-nodes) of the edge
        """
        speed_limit = net_params.additional_params["speed_limit"]

        edges = [
            {
                "id": "edge0", 
                "from": "left0", 
                "to": "middle0", 
                "length": 250, 
                "spreadType": "center",
                "numLanes": 6, # get from params
                "speed": speed_limit,},
            {
                "id": "edge1",
                "from": "middle0",
                "to": "middle1",
                "length": 500,
                "spreadType": "center",
                "numLanes": 4, # get from params
                "speed": speed_limit,},

            {
                "id": "edge2",
                "from": "middle1",
                "to": "middle2",
                "length": 500,
                "spreadType": "center",
                "numLanes": 3, # get from params
                "speed": speed_limit,},
            {
                "id": "edge3",
                "from": "middle2",
                "to": "right0",
                "length": 250,
                "spreadType": "center",
                "numLanes": 2, # get from params
                "speed": speed_limit,},

        ]

        return edges

    
    def specify_routes(self, net_params):
        """
        Routes: sequence of edges the vehicles can traverse before restarting
        There can either be a single route per egde or multiple routes per edge
        Additionally, there can also be per vehicle route
        
        Single routes: deterministic e.g., ring road
        Multiple routes: Shocastic, tupes with probability (The sum must be 1)
        """
        routes = {
            # key = edge a vehicle is on, value = list of edges the vehicle can traverse
            "edge0": ["edge0", "edge1", "edge2"],
            "edge1": ["edge1", "edge2", "edge3"],
            "edge2": ["edge2", "edge3"],
            "edge3": ["edge3" ],
        }

        return routes

    def specify_edge_starts(self):
        """
        Starting position of vehicles on the edge
        """
        edge_starts = [
            ("edge0", 0),
            ("edge1", 250),
            ("edge2", 750),
            ("edge3", 1250),
        ]

        return edge_starts

    def specify_connections(self, net_params):
        """
        Connections: Specify how lanes are connected to each other
        """
        conn_dic = {}
        conn = []
        
        # Map the 6 lanes to 4 lanes
        # Outer lanes merge inwards
        lane_mapping_1 = {0:0, 
                          1:0, 
                          2:2, 
                          3:3, 
                          4:3, 
                          5:3} 

        # Connect lanes from edge0 to edge1
        for i in range(6):
            conn += [{
                "from": "edge0",
                "to": "edge1",
                "fromLane": i,
                "toLane": lane_mapping_1[i]
            }]
        conn_dic["edge2"] = conn
        
        conn = []
        
        # Map the 4 lanes to 3 lanes
        # The 2 inner lanes merge inwards to the middle lane
        lane_mapping_2 = {0:0, 
                          1:1, 
                          2:1, 
                          3:2}

        # Connect lanes from edge1 to edge2
        for i in range(4):
            conn += [{
                "from": "edge1",
                "to": "edge2",
                "fromLane": i,
                "toLane": lane_mapping_2[i]
            }]
        conn_dic["edge2"] = conn + conn_dic["edge2"]
            
        conn = []
        
        # Map the 3 lanes to 2 lanes
        # ??
        lane_mapping_3 = {0:0, 
                          1:1, 
                          2:1}

        # Connect lanes from edge2 to edge3
        for i in range(3):
            conn += [{
                "from": "edge2",
                "to": "edge3",
                "fromLane": i,
                "toLane": lane_mapping_3[i]
            }]
        conn_dic["edge3"] = conn
        return conn_dic

    def specify_centroids(self, net_params):
        """
        Centroids: specify where the vehicles are coming from and going
        """
        centroids = [] 

        centroids += [{
            "id": "1",
            "from": None,
            "to": "left0",
            "x": -30,
            "y": 0,
        }]
 
        centroids += [{
            "id": "3",
            "from": "right0",
            "to": None,
            "x": 1500+30,
            "y": 0,
        }]

    #     return centroids



