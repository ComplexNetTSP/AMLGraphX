API reference
=============

This reference documents the supported Python interface. Private modules and implementation-specific bindings are intentionally excluded. Public APIs may evolve while AMLGraphX is pre-1.0; prefer imports from the modules below rather than importing implementation files directly.

Datasets
--------

.. automodule:: amlgraphx.datasets
   :members:
   :show-inheritance:

Canonical data and snapshot preparation
----------------------------------------

.. automodule:: amlgraphx.data
   :members:
   :show-inheritance:

Graph preparation and PyTorch Geometric conversion
---------------------------------------------------

.. automodule:: amlgraphx.graph
   :members:
   :exclude-members: GraphSnapshot
   :show-inheritance:

Temporal split protocols
------------------------

.. automodule:: amlgraphx.split
   :members:
   :show-inheritance:

Tabular graph features
----------------------

.. automodule:: amlgraphx.tabular
   :members:
   :show-inheritance:

Classical baselines
-------------------

.. automodule:: amlgraphx.baselines
   :members:
   :show-inheritance:
