; Automatically converted to only require STRIPS and negative preconditions

(define (domain spanner)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (location ?x)
    (locatable ?x)
    (man ?x)
    (nut ?x)
    (spanner ?x)
    (at ?m ?l)
    (carrying ?m ?s)
    (useable ?s)
    (link ?l1 ?l2)
    (tightened ?n)
    (loose ?n)
  )

  (:action walk
    :parameters (?start ?end ?m)
    :precondition (and
      (location ?start)
      (location ?end)
      (man ?m)
      (at ?m ?start)
      (link ?start ?end)
    )
    :effect (and
      (not (at ?m ?start))
      (at ?m ?end)
    )
  )

  (:action pickup_spanner
    :parameters (?l ?s ?m)
    :precondition (and
      (location ?l)
      (spanner ?s)
      (man ?m)
      (at ?m ?l)
      (at ?s ?l)
    )
    :effect (and
      (not (at ?s ?l))
      (carrying ?m ?s)
    )
  )

  (:action tighten_nut
    :parameters (?l ?s ?m ?n)
    :precondition (and
      (location ?l)
      (spanner ?s)
      (man ?m)
      (nut ?n)
      (at ?m ?l)
      (at ?n ?l)
      (carrying ?m ?s)
      (useable ?s)
      (loose ?n)
    )
    :effect (and
      (not (loose ?n))
      (not (useable ?s))
      (tightened ?n)
    )
  )
)
