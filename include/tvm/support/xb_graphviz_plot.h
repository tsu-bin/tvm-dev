#ifndef XB_GRAPHVIZ_PLOT_H_
#define XB_GRAPHVIZ_PLOT_H_

#include <boost/graph/graphviz.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/subgraph.hpp>
#include <iostream>


using GraphvizAttributes = 
    std::map<std::string, std::string>;

using Graph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, 
        boost::property<boost::vertex_attribute_t, GraphvizAttributes>,
        boost::property<boost::edge_index_t, int, boost::property<boost::edge_attribute_t, GraphvizAttributes> >,
        boost::property<boost::graph_name_t, std::string,
        boost::property<boost::graph_graph_attribute_t,  GraphvizAttributes,
        boost::property<boost::graph_vertex_attribute_t, GraphvizAttributes,
        boost::property<boost::graph_edge_attribute_t,   GraphvizAttributes>
        > > >
    >;
using SubGraph = boost::subgraph<Graph>;
using Vertex = Graph::vertex_descriptor;


class MyCanvas;
class MySubGraph;

class MyVertex {
    SubGraph* root_graph_ = nullptr;
    Vertex root_vtx_;
    SubGraph* parrent_subgraph_ = nullptr;
    Vertex subgraph_vtx_;

friend MyCanvas;
friend MySubGraph;
friend void add_edge(const MyVertex& src, const MyVertex& tgt, const std::string& label);

public:
    void update_label(const std::string& label){
        // put( get(boost::vertex_attribute, parrent_subgraph_ ? *parrent_subgraph_ : *root_graph_), 
        //         (parrent_subgraph_ ? subgraph_vtx_ : root_vtx_), 
        //         GraphvizAttributes{ {"label", label} }
        //     );

        get(boost::vertex_attribute, parrent_subgraph_ ? *parrent_subgraph_ : *root_graph_)
        [(parrent_subgraph_ ? subgraph_vtx_ : root_vtx_)]["label"] = label;
    }

    void append_label(const std::string& append_content){
        std::string& label_to_append = get(boost::vertex_attribute, parrent_subgraph_ ? *parrent_subgraph_ : *root_graph_)
        [(parrent_subgraph_ ? subgraph_vtx_ : root_vtx_)]["label"];

        label_to_append.append(append_content);
    }
};


class MySubGraph {
    SubGraph* rootgraph_ = nullptr;
    SubGraph* subgraph_ = nullptr;
    MyVertex label_vtx_;

friend MyCanvas;

public:
    MyVertex add_vtx(const std::string& label, 
            const std::string& overwrite_shape = "",
            const std::string& overwrite_fillcolor = "") {
        MyVertex ret;
        ret.root_graph_ = rootgraph_;
        ret.root_vtx_ = add_vertex(*rootgraph_);
        ret.parrent_subgraph_ = subgraph_;
        ret.subgraph_vtx_ = add_vertex(ret.root_vtx_, *subgraph_);

        GraphvizAttributes attrs;
        attrs["label"] = label;
        if (not overwrite_shape.empty()) {
            attrs["shape"] = overwrite_shape;
        }
        if (not overwrite_fillcolor.empty()) {
            attrs["fillcolor"] = overwrite_fillcolor;
            attrs["style"] = "filled";
        }
        put( get(boost::vertex_attribute, *subgraph_), ret.subgraph_vtx_, attrs );

        return ret;
    }

    operator MyVertex() {
        return label_vtx_;
    }
};


class MyCanvas {
    std::unique_ptr<SubGraph> rootgraph_;
public:
    static MyCanvas create() {
        MyCanvas ret;
        ret.rootgraph_ = std::make_unique<SubGraph>();
        auto& attr_map = get_property(*(ret.rootgraph_), boost::graph_vertex_attribute);
        attr_map["shape"] = "Mrecord";
        attr_map["fillcolor"] = "powderblue";
        attr_map["style"] = "filled";
        return ret;
    }

    void write_to(const std::string& output_file){
        std::ofstream os(output_file);
        write_graphviz(os, *rootgraph_);
    }

    MyVertex add_vtx(const std::string& label, 
            const std::string& overwrite_shape = "",
            const std::string& overwrite_fillcolor = "") {
        MyVertex ret;
        ret.root_graph_ = rootgraph_.get();
        ret.root_vtx_ = add_vertex(*rootgraph_);

        GraphvizAttributes attrs;
        attrs["label"] = label;
        if (not overwrite_shape.empty()) {
            attrs["shape"] = overwrite_shape;
        }
        if (not overwrite_fillcolor.empty()) {
            attrs["fillcolor"] = overwrite_fillcolor;
            attrs["style"] = "filled";
        }
        put( get(boost::vertex_attribute, *rootgraph_), ret.root_vtx_, attrs);

        return ret;
    }

    MySubGraph create_subgraph(const std::string& label) {
        static int subgraph_idx = 0;
        SubGraph& sub = rootgraph_->create_subgraph();
        get_property(sub, boost::graph_name) = "cluster_"+std::to_string(subgraph_idx++);
        get_property(sub, boost::graph_graph_attribute)["bgcolor"] = "gray90";
        // get_property(sub, boost::graph_graph_attribute)["style"] = "filled";
        // get_property(sub, boost::graph_graph_attribute)["bgcolor"] = "transparent";
        get_property(sub, boost::graph_graph_attribute)["color"] = "sienna";
        get_property(sub, boost::graph_graph_attribute)["penwidth"] = "2.5";

        get_property(sub, boost::graph_vertex_attribute) = 
            GraphvizAttributes{{"shape","Rectangle"}, {"fillcolor", "palegreen2"}, {"style","filled"}};

        MySubGraph ret;
        ret.rootgraph_ = rootgraph_.get();
        ret.subgraph_ = &sub;
        ret.label_vtx_ = ret.add_vtx(label, "Mrecord", "powderblue");

        return ret;
    }
};


inline void add_edge(const MyVertex& src, const MyVertex& tgt, const std::string& label = "") {
    SubGraph* scope = nullptr;
    Vertex s_v, t_v;
    if (src.parrent_subgraph_ && (src.parrent_subgraph_ == tgt.parrent_subgraph_) ) {
        scope = src.parrent_subgraph_;
        s_v = src.subgraph_vtx_, t_v = tgt.subgraph_vtx_;
    }
    else {
        scope = src.root_graph_;
        s_v = src.root_vtx_, t_v = tgt.root_vtx_;
    }

    auto [edge, _] = add_edge(s_v, t_v, *scope);
    if (not label.empty()) {
        put( get(boost::edge_attribute, *scope), 
                edge, 
                GraphvizAttributes{ {"label", label}, {"penwidth", "2.0"} }
            );
    }
}

#endif  // XB_GRAPHVIZ_PLOT_H_