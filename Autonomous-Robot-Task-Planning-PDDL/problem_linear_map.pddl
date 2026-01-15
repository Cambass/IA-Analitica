(define (problem mapa-lineal)
  (:domain limpieza-aulas)

  (:objects
    wall-e - robot
    pasillo aula1 aula2 aula3 residuos - ubicacion
    bolsa1 - bolsa
  )

  (:init
    (en-robot wall-e pasillo)
    (mano-vacia wall-e)

    ;; MAPA MODIFICADO (Lineal / Conexión interna)
    ;; Pasillo conecta a Residuos, Aula1 y Aula3
    (conectado pasillo residuos) (conectado residuos pasillo)
    (conectado pasillo aula1)    (conectado aula1 pasillo)
    (conectado pasillo aula3)    (conectado aula3 pasillo)
    
    ;; PERO... Aula 2 solo es accesible desde Aula 1 (No desde pasillo)
    (conectado aula1 aula2)      (conectado aula2 aula1)
    
    (es-cuarto-residuos residuos)

    ;; Solo una bolsa en la habitación más lejana (Aula 2)
    (en-bolsa bolsa1 aula2)
  )

  (:goal
    (and
      (en-bolsa bolsa1 residuos)
      (aula-limpia aula1)
      (aula-limpia aula2)
      (aula-limpia aula3)
    )
  )
)