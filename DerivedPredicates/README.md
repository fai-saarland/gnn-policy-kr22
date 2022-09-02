<h1>Implements state augmentation with derived predicates</h1>

Derived predicates are calculated with DATALOG rules until fixpoint is reached.
The rules are applied to facts (static atoms) and to set of atoms for each
state. The resulting (augmented) problem, comprising augmened set of facts
and augmented states, is written to disk.

The set of rules are indexed with the ```domain``` parameter. The rules are
stored in the registry (default is ```registry_rules.json```). It can be modified
or extended.

<h3>Examples</h3>

For augmenting the problem in the file ```../Data/states/train/blocks/probBLOCKS-4-0.states```
with the set of rules for ```blocks```, execute:

```
python3 augment_states_with_derived_predicates.py ../Data/states/train/blocks/probBLOCKS-4-0.states blocks
```

The augmented problem is stored in ```../Data/states/train/blocks/probBLOCKS-4-0.states.augmented```.
For augmenting all files with suffix ```.states``` in folder ```../Data/states/train/blocks```
with the set of rules for ```blocks```, execute:

```
python3 augment_states_with_derived_predicates.py ../Data/states/train/blocks blocks
```

This is equivalent to calling ```augment_states_with_derived_predicates.py``` for each
```.states``` file in folder ```../Data/states/train/blocks```.

For augmenting all files with suffix ```.states``` in folder or subfolders in ```../Data/states/train```,
where the domain for each file is determined by its path, execute:

```
python3 augment_states_with_derived_predicates.py ../Data/states/train --recursive
```

<h3>Rules</h3>

Currently, the registry has the following sets of rules:

* ```blocks```: transitive closure of ```on/2``` as ```above/2```
* ``` logistics```: define role compositions for:
  - at<sub>G</sub> &#8728; in-city
  - at &#8728; in-city
  - in &#8728; at &#8728; in-city
  - in &#8728; at
* ```spanner```: transitve closure of ```link/2``` as ```link+/2```


