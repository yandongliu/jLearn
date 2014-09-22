package learning.classifier.tree;

/**
 * Created by yandong on 7/11/14.
 */
public class TreeNode<T> {
    enum NodeType {
        FEATURE, TERMINAL
    }
    String featureName;
    int featureId;
    T val;
    NodeType type;
    double cut;
    TreeNode leftChild, rightChild;
    public TreeNode(String name, int id, Double cut) {
        this.type = NodeType.FEATURE;
        this.featureName = name;
        this.featureId = id;
        this.cut = cut;
    }
    public TreeNode(String name, int id, T val) {
        this.featureName="";
        this.type = NodeType.TERMINAL;
        this.val = val;
    }
    public void setLeftChild(TreeNode node) {
        this.leftChild = node;
    }
    public void setRightChild(TreeNode node) {
        this.rightChild = node;
    }
}
