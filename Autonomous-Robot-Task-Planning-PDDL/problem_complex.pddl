(define (problem limpieza-extrema)
  (:domain limpieza-aulas)

  (:objects
    wall-e - robot
    pasillo aula1 aula2 aula3 residuos - ubicacion
    bolsa1 bolsa2 bolsa3 bolsa4 - bolsa
  )

  (:init
    ;; Robot empieza en residuos (descargado)
    (en-robot wall-e residuos)
    (mano-vacia wall-e)

    ;; Mapa (Igual al original)
    (conectado pasillo aula1) (conectado aula1 pasillo)
    (conectado pasillo aula2) (conectado aula2 pasillo)
    (conectado pasillo aula3) (conectado aula3 pasillo)
    (conectado pasillo residuos) (conectado residuos pasillo)
    (es-cuarto-residuos residuos)

    ;; DISTRIBUCIÓN DE BASURA (Más compleja)
    (en-bolsa bolsa1 aula1)
    (en-bolsa bolsa2 aula1) ; Dos bolsas en el aula 1
    (en-bolsa bolsa3 aula2)
    (en-bolsa bolsa4 aula3)
  )

  (:goal
    (and
      ;; Todas las bolsas a la basura
      (en-bolsa bolsa1 residuos)
      (en-bolsa bolsa2 residuos)
      (en-bolsa bolsa3 residuos)
      (en-bolsa bolsa4 residuos)
      ;; Todo limpio
      (aula-limpia aula1)
      (aula-limpia aula2)
      (aula-limpia aula3)
    )
  )
)