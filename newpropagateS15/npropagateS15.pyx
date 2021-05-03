
import numpy as np







cpdef npropagateS15(double[:,:,:] H3real, double[:,:,:] H3imag, double[:] rhoreal , double[:] rhoimag, double dt, int lenz, int nrec, double[:,:] rho_out):
    
    cdef int sz = 9
    cdef int x,y,z,k = 0
    cdef double dth=0.5*dt
    cdef double dz = 0.0

    
    cdef double rhor[9]
    cdef double rhoi[9]

    cdef double tempar[9]
    cdef double tempai[9]
    cdef double tempbr[9]
    cdef double tempbi[9]
    cdef double temp1r[9]
    cdef double temp1i[9]

    cdef int index=0
    
    cdef double[:,:] rho_out_re = rho_out
    cdef double[:,:] rho_out_im = rho_out

    cpdef result = np.zeros((sz,nrec), dtype=np.complex128)
  
    
    for x in range(sz):
        rhor[x]=rhoreal[x]
        rhoi[x]=rhoimag[x]
        tempar[x]=dz
        tempai[x]=dz
        tempbr[x]=dz
        tempbi[x]=dz
        temp1r[x]=dz
        temp1i[x]=dz
        
    x=0

    for z in range(lenz):
        tempar[0]=dz
        tempai[0]=dz
                
        tempar[1]=H3real[z,1,0]*rhor[0]-H3imag[z,1,0]*rhoi[0]  + H3real[z,1,1]*rhor[1]-H3imag[z,1,1]*rhoi[1]
        tempai[1]=H3imag[z,1,0]*rhor[0]+H3real[z,1,0]*rhoi[0]  + H3imag[z,1,1]*rhor[1]+H3real[z,1,1]*rhoi[1]
                
        tempar[2]=H3real[z,2,0]*rhor[0]-H3imag[z,2,0]*rhoi[0]  + H3real[z,2,2]*rhor[2]-H3imag[z,2,2]*rhoi[2]
        tempai[2]=H3imag[z,2,0]*rhor[0]+H3real[z,2,0]*rhoi[0]  + H3imag[z,2,2]*rhor[2]+H3real[z,2,2]*rhoi[2]
                
        tempar[3]=H3real[z,3,0]*rhor[0]-H3imag[z,3,0]*rhoi[0]  + H3real[z,3,3]*rhor[3]-H3imag[z,3,3]*rhoi[3]
        tempai[3]=H3imag[z,3,0]*rhor[0]+H3real[z,3,0]*rhoi[0]  + H3imag[z,3,3]*rhor[3]+H3real[z,3,3]*rhoi[3]

        tempar[4]=H3real[z,4,2]*rhor[2]-H3imag[z,4,2]*rhoi[2]  + H3real[z,4,3]*rhor[3]-H3imag[z,4,3]*rhoi[3] + H3real[z,4,4]*rhor[4]-H3imag[z,4,4]*rhoi[4]
        tempai[4]=H3imag[z,4,2]*rhor[2]+H3real[z,4,2]*rhoi[2]  + H3imag[z,4,3]*rhor[3]+H3real[z,4,3]*rhoi[3] + H3imag[z,4,4]*rhor[4]+H3real[z,4,4]*rhoi[4]

        tempar[5]=H3real[z,5,1]*rhor[1]-H3imag[z,5,1]*rhoi[1]  + H3real[z,5,3]*rhor[3]-H3imag[z,5,3]*rhoi[3] + H3real[z,5,5]*rhor[5]-H3imag[z,5,5]*rhoi[5]
        tempai[5]=H3imag[z,5,1]*rhor[1]+H3real[z,5,1]*rhoi[1]  + H3imag[z,5,3]*rhor[3]+H3real[z,5,3]*rhoi[3] + H3imag[z,5,5]*rhor[5]+H3real[z,5,5]*rhoi[5]

        tempar[6]=H3real[z,6,1]*rhor[1]-H3imag[z,6,1]*rhoi[1]  + H3real[z,6,2]*rhor[2]-H3imag[z,6,2]*rhoi[2] + H3real[z,6,6]*rhor[6]-H3imag[z,6,6]*rhoi[6]
        tempai[6]=H3imag[z,6,1]*rhor[1]+H3real[z,6,1]*rhoi[1]  + H3imag[z,6,2]*rhor[2]+H3real[z,6,2]*rhoi[2] + H3imag[z,6,6]*rhor[6]+H3real[z,6,6]*rhoi[6]

        tempar[7]=H3real[z,7,4]*rhor[4]-H3imag[z,7,4]*rhoi[4]  + H3real[z,7,5]*rhor[5]-H3imag[z,7,5]*rhoi[5] + H3real[z,7,6]*rhor[6]-H3imag[z,7,6]*rhoi[6] + H3real[z,7,7]*rhor[7]-H3imag[z,7,7]*rhoi[7]
        tempai[7]=H3imag[z,7,4]*rhor[4]+H3real[z,7,4]*rhoi[4]  + H3imag[z,7,5]*rhor[5]+H3real[z,7,5]*rhoi[5] + H3imag[z,7,6]*rhor[6]+H3real[z,7,6]*rhoi[6] + H3imag[z,7,7]*rhor[7]+H3real[z,7,7]*rhoi[7]

        tempar[8]=H3real[z,8,4]*rhor[4]-H3imag[z,8,4]*rhoi[4]  + H3real[z,8,5]*rhor[5]-H3imag[z,8,5]*rhoi[5] + H3real[z,8,6]*rhor[6]-H3imag[z,8,6]*rhoi[6] + H3real[z,8,8]*rhor[8]-H3imag[z,8,8]*rhoi[8]
        tempai[8]=H3imag[z,8,4]*rhor[4]+H3real[z,8,4]*rhoi[4]  + H3imag[z,8,5]*rhor[5]+H3real[z,8,5]*rhoi[5] + H3imag[z,8,6]*rhor[6]+H3real[z,8,6]*rhoi[6] + H3imag[z,8,8]*rhor[8]+H3real[z,8,8]*rhoi[8] 

        for x in range(sz):
            temp1r[x]=tempar[x]*dt+rhor[x]
            temp1i[x]=tempai[x]*dt+rhoi[x]
        
        
        if z == (lenz-1):
            tempbr[0]=dz
            tempbi[0]=dz
            tempbr[1]=dz
            tempbi[1]=dz
            tempbr[2]=dz
            tempbi[2]=dz
            tempbr[3]=dz
            tempbi[3]=dz
            tempbr[4]=dz
            tempbi[4]=dz
            tempbr[5]=dz
            tempbi[5]=dz
            tempbr[6]=dz
            tempbi[6]=dz
            tempbr[7]=dz
            tempbi[7]=dz
            tempbr[8]=dz
            tempbi[8]=dz

        else:
            tempbr[0]=dz
            tempbi[0]=dz
        
            tempbr[1]=H3real[z+1,1,0]*temp1r[0]-H3imag[z+1,1,0]*temp1i[0]  + H3real[z+1,1,1]*temp1r[1]-H3imag[z+1,1,1]*temp1i[1]
            tempbi[1]=H3imag[z+1,1,0]*temp1r[0]+H3real[z+1,1,0]*temp1i[0]  + H3imag[z+1,1,1]*temp1r[1]+H3real[z+1,1,1]*temp1i[1]
                
            tempbr[2]=H3real[z+1,2,0]*temp1r[0]-H3imag[z+1,2,0]*temp1i[0]  + H3real[z+1,2,2]*temp1r[2]-H3imag[z+1,2,2]*temp1i[2]
            tempbi[2]=H3imag[z+1,2,0]*temp1r[0]+H3real[z+1,2,0]*temp1i[0]  + H3imag[z+1,2,2]*temp1r[2]+H3real[z+1,2,2]*temp1i[2]
                
            tempbr[3]=H3real[z+1,3,0]*temp1r[0]-H3imag[z+1,3,0]*temp1i[0]  + H3real[z+1,3,3]*temp1r[3]-H3imag[z+1,3,3]*temp1i[3]
            tempbi[3]=H3imag[z+1,3,0]*temp1r[0]+H3real[z+1,3,0]*temp1i[0]  + H3imag[z+1,3,3]*temp1r[3]+H3real[z+1,3,3]*temp1i[3]

            tempbr[4]=H3real[z+1,4,2]*temp1r[2]-H3imag[z+1,4,2]*temp1i[2]  + H3real[z+1,4,3]*temp1r[3]-H3imag[z+1,4,3]*temp1i[3] + H3real[z+1,4,4]*temp1r[4]-H3imag[z+1,4,4]*temp1i[4]
            tempbi[4]=H3imag[z+1,4,2]*temp1r[2]+H3real[z+1,4,2]*temp1i[2]  + H3imag[z+1,4,3]*temp1r[3]+H3real[z+1,4,3]*temp1i[3] + H3imag[z+1,4,4]*temp1r[4]+H3real[z+1,4,4]*temp1i[4]

            tempbr[5]=H3real[z+1,5,1]*temp1r[1]-H3imag[z+1,5,1]*temp1i[1]  + H3real[z+1,5,3]*temp1r[3]-H3imag[z+1,5,3]*temp1i[3] + H3real[z+1,5,5]*temp1r[5]-H3imag[z+1,5,5]*temp1i[5]
            tempbi[5]=H3imag[z+1,5,1]*temp1r[1]+H3real[z+1,5,1]*temp1i[1]  + H3imag[z+1,5,3]*temp1r[3]+H3real[z+1,5,3]*temp1i[3] + H3imag[z+1,5,5]*temp1r[5]+H3real[z+1,5,5]*temp1i[5]

            tempbr[6]=H3real[z+1,6,1]*temp1r[1]-H3imag[z+1,6,1]*temp1i[1]  + H3real[z+1,6,2]*temp1r[2]-H3imag[z+1,6,2]*temp1i[2] + H3real[z+1,6,6]*temp1r[6]-H3imag[z+1,6,6]*temp1i[6]
            tempbi[6]=H3imag[z+1,6,1]*temp1r[1]+H3real[z+1,6,1]*temp1i[1]  + H3imag[z+1,6,2]*temp1r[2]+H3real[z+1,6,2]*temp1i[2] + H3imag[z+1,6,6]*temp1r[6]+H3real[z+1,6,6]*temp1i[6]

            tempbr[7]=H3real[z+1,7,4]*temp1r[4]-H3imag[z+1,7,4]*temp1i[4]  + H3real[z+1,7,5]*temp1r[5]-H3imag[z+1,7,5]*temp1i[5] + H3real[z+1,7,6]*temp1r[6]-H3imag[z+1,7,6]*temp1i[6] + H3real[z+1,7,7]*temp1r[7]-H3imag[z+1,7,7]*temp1i[7]
            tempbi[7]=H3imag[z+1,7,4]*temp1r[4]+H3real[z+1,7,4]*temp1i[4]  + H3imag[z+1,7,5]*temp1r[5]+H3real[z+1,7,5]*temp1i[5] + H3imag[z+1,7,6]*temp1r[6]+H3real[z+1,7,6]*temp1i[6] + H3imag[z+1,7,7]*temp1r[7]+H3real[z+1,7,7]*temp1i[7]

            tempbr[8]=H3real[z+1,8,4]*temp1r[4]-H3imag[z+1,8,4]*temp1i[4]  + H3real[z+1,8,5]*temp1r[5]-H3imag[z+1,8,5]*temp1i[5] + H3real[z+1,8,6]*temp1r[6]-H3imag[z+1,8,6]*temp1i[6] + H3real[z+1,8,8]*temp1r[8]-H3imag[z+1,8,8]*temp1i[8]
            tempbi[8]=H3imag[z+1,8,4]*temp1r[4]+H3real[z+1,8,4]*temp1i[4]  + H3imag[z+1,8,5]*temp1r[5]+H3real[z+1,8,5]*temp1i[5] + H3imag[z+1,8,6]*temp1r[6]+H3real[z+1,8,6]*temp1i[6] + H3imag[z+1,8,8]*temp1r[8]+H3real[z+1,8,8]*temp1i[8]

            
        for x in range(sz):
                 
            rhor[x] = dth*(tempar[x] + tempbr[x]) + rhor[x]
            rhoi[x] = dth*(tempai[x] + tempbi[x]) + rhoi[x]
        

        if z > (lenz-nrec-1):
            for x in range(sz):
                rho_out_re[x,index]=rhor[x]
                rho_out_im[x,index]=rhoi[x]
            index = index +1

        
            
    result.real=rho_out_re
    result.imag=rho_out_im
    return result

