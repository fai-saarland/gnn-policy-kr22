(define (problem swapping-2)
    (:domain swapping)
    (:objects e0 p0 e1 p1)
    (:init
        (element e0)
        (position p0)
        (element e1)
        (position p1)
        (at e1 p0)
        (at e0 p1)
    )
    (:goal
        (and
            (at e0 p0)
            (at e1 p1)
        )
    )
)
