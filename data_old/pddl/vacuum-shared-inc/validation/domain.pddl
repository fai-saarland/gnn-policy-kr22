(define (domain robot-vacuum)
  (:requirements :strips :negative-preconditions)
  (:predicates (at ?r ?l) (dirty ?l) (adjacent ?r ?l1 ?l2))

  (:action drive
   :parameters (?r ?l1 ?l2)
   :precondition (and (at ?r ?l1) (adjacent ?r ?l1 ?l2))
   :effect (and (not (at ?r ?l1)) (at ?r ?l2)))

  (:action clean
   :parameters (?r ?l)
   :precondition (and (dirty ?l) (at ?r ?l))
   :effect (and (not (dirty ?l))))
)
