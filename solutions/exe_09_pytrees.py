import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Jax's PyTrees

    ## Lesson Goals:

    By the end of this lesson, you'll know to use the `PyTree` structure and its associated methods to keep your code clean; mastering these concepts and the autodiff will set us up to build our own simple neural network, _a la_  [Equinox](https://github.com/patrick-kidger/equinox)

    ## Core Concepts:

    - What is a `PyTree`?
    - what functions does Jax provide to interact with PyTrees? 
    - Registering `dataclass`-es to leverage the `PyTree` ecosystem

    ## Concepts In action:
    """
    )
    return


@app.cell
def _():
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass

    import jax
    import jax.numpy as jnp
    from jax import random
    from jax import make_jaxpr
    import marimo as mo
    return dataclass, jax, jnp, mo, np


@app.cell
def _(jax, jnp):
    def analyze_pytrees(*args, verbose_tree_print=False):
        for tree_idx, pytree in enumerate(args, start=1):
            leaves = jax.tree.leaves(pytree)
            if verbose_tree_print:
                print(f'Pytree: {repr(pytree):<30}')
            print(f'Tree: {tree_idx}')
            for i, leaf in enumerate(leaves, start=1):
                print(f'\tLeaf #: {i}:')
                print(f'\tLeaf Type: {type(leaf)}')
                if isinstance(leaf, jnp.ndarray) and len(leaf.shape) >= 2:
                    builder = '\t'
                    for row in leaf:
                        for el in row:
                            builder = builder + str(el) + ','
                        builder = builder + '\n\t\t'
                    leaf = builder
                print(f'\tLeaf val: {leaf}')
                print('\t' + '*' * 20)
            print('\n')
    return (analyze_pytrees,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # What is a PyTree

    A PyTree is a nested structure composed of objects. Much like a tree (in computer science) it can be broken down into nodes: leaf and non-leaf nodes. A leaf node can be thought of as a container that has been "registered" with `Jax` and cannot be deconstructed more, while a leaf is anything else; `Jax` deconstructs things like lists and dictionaries into individual elements, but keeps strings as they are. Here are some practical examples that should cover most data types you would use with `Jax`

    Let's take a look at some samples. Let's count how many leaves there are here:

    ```python
    conv1 = {
        "device": "CPU",
        "kernel": jnp.asarray(np.eye(5)),
        "bias": jnp.asarray(np.ones(5)),
        "indices": [1,2,"3"]
    }
    ```

    and here:

    ```python
    conv2 = {
        "device": "GPU",
        "kernel": jnp.asarray(np.eye(5)),
        "bias": jnp.asarray(np.ones(5)),
        "metadata": {"gpu": 0, "dtype": jnp.float64}
    }
    ```
    """
    )
    return


@app.cell
def _(jnp, np):
    dense1 = {
        "device": "CPU",
        "W": jnp.asarray(np.eye(5)),
        "b": jnp.asarray(np.ones(5)),
    }

    dense2 = {
        "device": "GPU",
        "W": jnp.asarray(np.eye(5)),
        "b": jnp.asarray(np.ones(5)),
        "metadata": {"gpu": 0, "dtype": jnp.float64}
    }
    return dense1, dense2


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Manually Counting Leaves

    Let's first count leaves to get an intuition
    """
    )
    return


@app.cell
def _(analyze_pytrees, dense1):
    analyze_pytrees(dense1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In `dense1` we had 6 elements:

    - (1)the bias, a jax vector

    - (2) the "device", a string

    - (3,4,5) the contents of the indices
    - (6), the kernel, the jax matrix

    Try to reason through `dense2` in the next cell on your own and verify your understanding - it's imperative that things click now before we move on.
    """
    )
    return


@app.cell
def _(analyze_pytrees, dense2):
    analyze_pytrees(dense2)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Exploring the `PyTree`

    Let's use some built-in functions to explore the `dense2` value
    """
    )
    return


@app.cell
def _(dense2, jax):
    def list_flattened(tree):
        flat_vals, flat_tree_def = jax.tree.flatten(tree)

        print(flat_vals)
        print(flat_tree_def)

    list_flattened(dense2)
    return


@app.cell
def _(dense2, jax):
    def list_leaves(tree):
        leaf_vals = jax.tree.leaves(tree)

        print(leaf_vals)

    list_leaves(dense2)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Takeaway: Leaves and Flatten

    `leaves` and `flatten` both return the flattened values as a list, but `flatten` also returns the tree definition, which we can use to reconstruct the original tree.

    **Note**: there are two variants of `leaves` and `flatten`: `x_with_path`, which breaks down the nested structure and generates the traversal path.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Reconstructing the PyTree

    Let's now take a look at using the result of `flatten` to reconstruct the tree
    """
    )
    return


@app.cell
def _(dense2, jax):
    def reconstruct_tree(in_tree):
        flat_vals, flat_tree_def = jax.tree.flatten(in_tree)
        recreated_tree = jax.tree.unflatten(flat_tree_def, flat_vals)
        print(recreated_tree.keys())
        print(recreated_tree["metadata"])
        print(f"\nWeights: {recreated_tree['W']}")
        print(f"\nBias: {recreated_tree['b']}")

    reconstruct_tree(dense2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Essential `PyTree` methods

    When we work with ML tasks, we often want to work with the leaves of our data structure, so it makes sense that the `Jax` team created utility functions to work with these structures. Let's explore a few of them.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## PyTree Definition""")
    return


@app.cell
def _(jnp, np):
    NUM_FEATURES = 4
    HIDDEN_DIM = 16


    PTDenseLayer = dict[str, 
        jnp.ndarray | 
        dict[str, str | int | jnp.dtype]
    ]
    def conv_pytree_constructor(
        W: jnp.ndarray,
        b: jnp.ndarray,
        metadata: dict[str, int]
    ) -> PTDenseLayer:
        return {
            "numerical": {
                "W": W,
                "b": b,
            },
            "metadata": {k:v for k,v in metadata.items()}
        }

    metadata = {"device": "GPU", "gpu": 0, "dtype": jnp.float64}
    dense_pt = conv_pytree_constructor(
        W=jnp.asarray(np.random.rand(NUM_FEATURES, HIDDEN_DIM)),
        b= jnp.asarray([1, 1, 1, 1]),
        metadata=metadata
    )

    return HIDDEN_DIM, NUM_FEATURES, dense_pt, metadata


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Associated `PyTree` methods

    We've covered a few functions to destructure the reconstruct the `PyTree` so far. Now let's discuss some of the methods in the context of the pre-defined `DenseLayer` and `PTDenseLayer`. If you've done functional programming before, some of these methods will look familiar to you:

    ```
    jax.tree.map
    jax.tree.reduce
    jax.tree.transpose

    jax.tree.map_with_path
    jax.tree.leaves_with_path
    jax.tree.flatten_with_path
    ```

    For this tutorial we focus on: `map` and `map_with_path`, which are essential for the datascience workflow.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Map

    A tree map works much like a standard map, where we apply a function to every leaf in our tree
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Simple Map

    Let's add two `PTDenseLayer` using

    ```python
    @jax.jit
    def add_numerics(v1, v2):
        return v1 + v2
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Reminder:

    As a reminder, here are the leaves in our `dense_pt`, only a few of which it makes sense to add together
    """
    )
    return


@app.cell
def _(analyze_pytrees, dense_pt):
    analyze_pytrees(dense_pt)
    return


@app.cell
def _(analyze_pytrees, dense_pt, jax):
    @jax.jit
    def add_numerics(v1, v2):
        return v1 + v2

    analyze_pytrees(dense_pt)
    print("\n After Addition")
    analyze_pytrees(jax.tree.map(
        # TODO: fill in the args for the `tree.map`
        add_numerics, dense_pt["numerical"], dense_pt["numerical"]
    ))
    return (add_numerics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Map-with-Path

    If you look at the previous cell, you'll notice that we had no notion of the nodes that we traversed to get to our leaf nodes. When is this information useful? Say you wanted to have some custom logic depending on the path. Let's explore this in the next example, where we initialize parameters of our dense layer for our simple neural network library.
    """
    )
    return


@app.cell
def _(jnp):
    params = {
        'network': {
            'metadata': {"device": "GPU", "gpu": 0, "dtype": jnp.float64},

            # Say we wanted custom initialization of parameters, depending on what layer we're on. 
            # A contrived example would be initialization based on where in the network (depth-wise) it is
            'layers': {
                0: {'dense': {'W': jnp.ones((128, 256)), "init": "Xavier", 'b': jnp.ones(256)}},
                1: {'dense': {'W': jnp.ones((256, 128)), "init": "Normal", 'b': jnp.ones(128)}}
            }
        }
    }
    return (params,)


@app.cell
def _(jax, jnp, params):
    def join_path(path: tuple):
        """
        Utility function to generate the entire path that we traverse
        """
        return '.'.join(str(key.key) if hasattr(key, 'key') else str(key) for key in path)


    def layer_specific_init(path: tuple, params: dict) -> dict:
        """Initialization here depends on where (layer-wise) the data is."""
        if not isinstance(params, dict) or "dense" not in params:
            return params

        path_str = join_path(path)
        param = params["dense"]

        # Extract layer number from path
        layer_num = None
        for key in path:
            key_str = str(key.key) if hasattr(key, 'key') else str(key)
            if key_str.isdigit():
                layer_num = int(key_str)
                break

        if layer_num is None:
            raise ValueError(f"Could not find layer number in path: {path_str}")

        # Different initialization based on layer depth
        if layer_num == 0:  
            # First layer - more conservative. Realistically not what we 
            # should do in prod, but illustrates our point.
            W = jax.random.normal(jax.random.PRNGKey(42), param["W"].shape) * 0.001
            init_type = "Small_Normal"
        elif layer_num == 1:  # Second layer - Xavier
            fan_in, fan_out = param["W"].shape
            bound = jnp.sqrt(6.0 / (fan_in + fan_out))
            W = jax.random.uniform(
                jax.random.PRNGKey(42), param["W"].shape, minval=-bound, maxval=bound
            )
            init_type = "Xavier"
        else:  # Deeper layers - He initialization
            fan_in = param["W"].shape[0]
            std = jnp.sqrt(2.0 / fan_in)
            W = jax.random.normal(jax.random.PRNGKey(42), param["W"].shape) * std
            init_type = "He"

        b = jnp.zeros_like(param["b"])
        return {"dense": {"W": W, "b": b, "init": init_type}}

    # This requires map_with_path because we need to know which layer we're in!
    jax.tree.map_with_path(
        layer_specific_init, 
        params,
        is_leaf=lambda x: isinstance(x, dict) and "dense" in x
    )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Registering Custom Objects

    We often represent data in the form of classes or dataclasses, if only for encapsulation of logic. Unfortunately, `Jax` doesn't work natively with dataclasses. Fortunately, the team has exposed methods to integrate dataclasses into the ecosystem.

    We will wrap up this lecture with our definition of `DenseLayer`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Dataclass Definition""")
    return


@app.cell
def _(dataclass, jax, jnp, np):

    @dataclass
    class DenseLayer:
        """
        The equivalent of conv1, structure-wise
        """
        numerical: dict[str, jnp.ndarray]
        metadata: dict[str, str]

        def __init__(self, W, b, metadata):
            self.numerical = {
                "W": W, "b": b
            }
            self.metadata = {k:v for k,v in metadata.items()}

        def __repr__(self):
            shapes = {k: v.shape for k,v in self.numerical.items()}
            return f"DL: Metadata: {self.metadata} with {shapes}"


    def verify_registration(initial: "RegisteredDenseLayer") -> bool:
        """
        Quick function to verify that the registration was correct and works
        """
        tree_vals, tree_defn = jax.tree.flatten(initial)
        reconstructed: "RegisteredDenseLayer" = jax.tree.unflatten(tree_defn, tree_vals)

        ####################################
        # Check Equality
        ####################################
        meta_keys_equal = set(initial.metadata.keys()) == set(reconstructed.metadata.keys())
        if not meta_keys_equal:
            print("Keys in our metadata were different!")
            return False
        meta_values_equal = True
        for k in initial.metadata.keys():
            meta_values_equal = meta_values_equal and (initial.metadata[k] == reconstructed.metadata[k])        

        w_equal = np.all(initial.numerical["W"] == reconstructed.numerical["W"])
        b_equal = np.all(initial.numerical["b"] == reconstructed.numerical["b"])

        print(f"Weights Equal: {w_equal}, Biases Equal: {b_equal}, Metadata Equal: {meta_values_equal}")
        return w_equal and b_equal and meta_keys_equal and meta_values_equal

    return DenseLayer, verify_registration


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Registering our `DenseLayer`

    1) Register the dense layer with `PyTree`

    2) Verify the flatten and unflatten with `verify_registration`

    ### Registration

    The registration uses [register_pytree_node](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_pytree_node.html#jax.tree_util.register_pytree_node), which has the following signature:

    ```python
    register_pytree_node(nodetype, flatten_func, unflatten_func, flatten_with_keys_func=None)
    \"""Parameters:

        nodetype (type[T]) – a Python type to register as a pytree.

        flatten_func (Callable[[T], tuple[_Children, _AuxData]]) – a function to be used during flattening, taking a value of type nodetype and returning a pair, with (1) an iterable for the children to be flattened recursively, and (2) some hashable auxiliary data to be stored in the treedef and to be passed to the unflatten_func.

        unflatten_func (Callable[[_AuxData, _Children], T]) – a function taking two arguments: the auxiliary data that was returned by flatten_func and stored in the treedef, and the unflattened children. The function should return an instance of nodetype.

        flatten_with_keys_func (Callable[[T], tuple[KeyLeafPairs, _AuxData]] | None)
    \"""
    ```
    """
    )
    return


@app.cell
def _(DenseLayer):
    from jax.tree_util import register_pytree_node

    class RegisteredDenseLayer(DenseLayer):
        def __repr__(self):
            shapes = {k: v.shape for k,v in self.numerical.items()}
            return f"RegisteredDenseLayer: Metadata: {self.metadata} with {shapes}"

    def dense_tree_flatten(dl: DenseLayer):
        md_keys = []
        md_values = []

        for k, v in dl.metadata.items():
            md_keys.append(k)
            md_values.append(v)

        children = tuple([dl.numerical["W"], dl.numerical["b"]] + md_values)
        aux_data = tuple(md_keys)
        return children, aux_data

    def dense_tree_unflatten(aux_data, children):
        W, b = children[0], children[1]
        reconstructed_metadata = {k: v for (k, v) in zip(aux_data, children[2:])}
        return RegisteredDenseLayer(W, b, reconstructed_metadata)


    register_pytree_node(
        RegisteredDenseLayer,
        dense_tree_flatten,    # tell JAX what are the children nodes
        dense_tree_unflatten   # tell JAX how to pack back into a 
    )
    return RegisteredDenseLayer, register_pytree_node


@app.cell
def _(
    HIDDEN_DIM,
    NUM_FEATURES,
    RegisteredDenseLayer,
    jnp,
    metadata,
    np,
    verify_registration,
):
    initial = RegisteredDenseLayer(
        W = jnp.asarray(np.random.rand(NUM_FEATURES, HIDDEN_DIM)),
        b= jnp.asarray([1, 1, 1, 1]),
        metadata=metadata
    )
    verify_registration(initial=initial)
    return (initial,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Revisiting an old friend....

    Let's look at the familiar function, `add_numerics`,

    ```python
    @jax.jit
    def add_numerics(v1, v2):
        return v1 + v2
    ```

    and try to run it!
    """
    )
    return


@app.cell
def _(add_numerics, initial, jax):
    print(initial.numerical)

    print(jax.tree.map(add_numerics, initial, initial, is_leaf=lambda x: isinstance(x, dict)).numerical)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The problem

    ### The API layer

    Unlike before, we can't easily slice our dictionary. In fact, we don't **want** to slice our dictionary - at the API level, our user shouldn't need to know what's happening under the hood and do all sorts of funky slicing.

    ### What went wrong?
    In our `dense_tree_flatten`, all of the `children` information was passed in and `jit` tried to do its thing; Jax flattened our data, as specified by us, and passed all of the information in the `children` variable to the `add_up` function. Ponder this statement as you solve the next exercise, where you make the `DenseLayer` compatible with `jit`.

    ### Fixing it

    Modify the definition below and fix the issue!
    """
    )
    return


@app.cell
def _(DenseLayer, RegisteredDenseLayer, register_pytree_node):
    class RegisteredDenseLayerCompatible(DenseLayer):
        def __repr__(self):
            shapes = {k: v.shape for k,v in self.numerical.items()}
            return f"RegisteredDenseLayer: Metadata: {self.metadata} with {shapes}"

    def dense_tree_flatten_compat(dl: DenseLayer):
        md_keys = []
        md_values = []

        for k, v in dl.metadata.items():
            md_keys.append(k)
            md_values.append(v)

        children = [dl.numerical["W"], dl.numerical["b"]] 
        return tuple(children), (md_keys, md_values)

    def dense_tree_unflatten_compat(aux_data, children):
        W, b = children
        reconstructed_metadata = {k: v for (k, v) in zip(*aux_data)}
        return RegisteredDenseLayer(W, b, reconstructed_metadata)


    register_pytree_node(
        RegisteredDenseLayerCompatible,
        dense_tree_flatten_compat,    # tell JAX what are the children nodes
        dense_tree_unflatten_compat   # tell JAX how to pack back into a 
    )
    return (RegisteredDenseLayerCompatible,)


@app.cell
def _(
    HIDDEN_DIM,
    NUM_FEATURES,
    RegisteredDenseLayerCompatible,
    add_numerics,
    jax,
    jnp,
    metadata,
    np,
):
    initial_dense_compat = RegisteredDenseLayerCompatible(
        W = jnp.asarray(np.random.rand(NUM_FEATURES, HIDDEN_DIM)),
        b= jnp.asarray([1, 1, 1, 1]),
        metadata=metadata
    )
    print(initial_dense_compat.numerical)

    print(jax.tree.map(add_numerics, initial_dense_compat, initial_dense_compat).numerical)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
