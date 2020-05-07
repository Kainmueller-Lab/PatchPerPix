INFO
=====

Overview
----------
main function consists of 5 major steps:
1. compute consensus array
2. rank patches by consensus
3. cover foreground with patches (external, by python)
4. compute affinity graph on patches
5. compute labeling by extracting connected components from graph or using
  mutex watershed (external, by python)


data and patch dimensions have to compiled in, e.g.

``` shell
	make DATAXSIZE=896 DATAYSIZE=720 DATAZSIZE=1 DATACSIZE=625 PSX=25 PSY=25 PSZ=1 TH=0.9 THI=0.1 NSX=50 NSY=50 NSZ=1
```

substeps can be computed individually, use makefile targets to create
executables for steps

``` shell
	make fillConsensus DATAXSIZE=896 DATAYSIZE=720 DATAZSIZE=1 DATACSIZE=625 PSX=25 PSY=25 PSZ=1 TH=0.9 THI=0.1 NSX=50 NSY=50 NSZ=1
```


Arguments
-----------
- affinities: path to predicted affinities file
- result\_folder: where to place output
- vote\_instances\_path: path to python vote_instances file
- inclSinglePatchCCS (0/1): use single patch ccs or not
