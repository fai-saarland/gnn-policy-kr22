(define (problem BLOCKS-8-1)
(:domain BLOCKS)
(:objects B A G C F D H E )
(:INIT (CLEAR E) (CLEAR H) (CLEAR D) (CLEAR F) (ONTABLE C) (ONTABLE G)
 (ONTABLE D) (ONTABLE F) (ON E C) (ON H A) (ON A B) (ON B G) (HANDEMPTY))
(:goal (AND (ON C D) (ON D B) (ON B G) (ON G F) (ON F H) (ON H A) (ON A E)))
)