The file to execute is execute.sh where you have to specify
-The file to analyse
-the variable we are interested in (gfp_nb,length_um,length_vtmvb,concentration,..)
-numarray is the number of time we parallely optimize the likelihood by
    starting from differents initial conditions [higer it is slower will be but
    potentially more precise]
-numproc is the number of times we divide the initial matrix in sub matrices
    [higher it is faster will be] rule of tumb: (total number of
    cells/numproc~10
-step is the time step for the prediction. The input equals 3 [min] but we can
    have prediction every step [min]
-dt is the time step [min] for computing the derivative (x[t+dt]-x[-dt])/2dt 
NB it might be necessary to modify pathprediction.sh or inferences.sh depending
on how much time/memory/.. is necessary for the current jobs running
NB the variable length_vbmvt = vetical_bottom-vertical_top has been created
THE OUTPUT
the only outputfile will be 
-var_ATHOS.csv where 
    -cell: is the unique identifier of a cell as in original file
    -d_var_dt: the derivative of the variable of interest
    -var_pred: the prediction from GPy
    -err: the variance on the prediction
    -var: the original input variable [every 3 min for MoMa]
    -time: the new time step
-allhyperinfer.txt
    the file containing all the value of the minimization

