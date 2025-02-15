# BlackJackDQL
## Ricardo Ibarra y Adrián López 
### Breve Explicación
El trabajo consiste en desarrollar el juego de BlackJack implementando DQL (Deep Q-Learning). Esto entrenando por base de aprendizaje por refuerzo, para así hacer que nuestro agente aprenda a tomar decisiones óptimas con el paso de los juegos. 

Se inicializa el agente, donde se define la red neuronal junto con una memoria que le sirve a nuestro agente como un conjunto de experiencias, para que estas experiencias le permitan mejorar conforme los juegos se vayan presentando.
Durante la simulación, a medida que nuestro agente toma decisiones, se registran lo que son las recompensas dependiendo del resultado de cada partida. Para mejorar su toma de decisiones, se aplica 'epsilon-greedy', que es un equilibrio de 'epsilon' que le permite explorar nuevas estrategtias y a la vez explotar loque ya ha aprendido. 

La red neuronal se entrena ajustando los valores de Q (recompensa esperada si se toma cierta acción específica) y, para mejorar la estabilidad del aprendizaje, se actualiza periódicamente una red de destino. Finalmente, los resultados de cada juego se registran y se analizan para evaluar el desempeño del agente, observando la evolución de su porcentaje de victorias, derrotas y empates.
