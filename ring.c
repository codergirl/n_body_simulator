/*
   Author: Sara Dadizadeh
   StudentID: 35746056
   Email: sara88@cs.ubc.ca
   
   This program is an simple simulation of the N-body problem.
   It takes two parameters,
       NUM_BODIES - the number of bodies to simulate
       NUM_TIME_STEPS - the number of times to repeat simulation
       
   Please run by going
       mpiexec -n NUM_THREADS ring NUM_BODIES NUM_TIME_STEPS

   I ran on the cs linux machines (okanagan/begbie) and the makefile works
   without alterations there.
   
   Please note that if NUM_BODIES does not divide evenly into NUM_THREADS,
   there will be elements at the end of the arrays which remain uninitialized.
   
   I use a ring to communicate the positions of the bodies. I wrap the 
   positions of the bodies. I also do not let the distance between bodies
   get too small, to prevent the forces from getting huge.
   
   Note. Every process has a copy of an array with the positions of all
   other bodies. My implementation retains consistent array indexing 
   across all the processes. The ordering of the particles however does 
   not matter, and this method reqiures extra calculations which aren't
   necessary. However it was easier to ensure that my messages were being
   passed correctly by implementing it this way.
   
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <mpe.h>
#include <mpe_graphics.h>
#include <unistd.h>

#define ROOT_PROCESS   0
#define WINDOW_SIZE    600
#define dt             0.1
#define WORLD          MPI_COMM_WORLD
#define TAG            1
#define CIRCLE_RAD     6

int NUM_BODIES = 8;
int NUM_TIME_STEPS = 1;

void initialize_arrays(float all_positions[][2], float v[][2], int chunk_size, int rank);
int calculate_force(int particle_pos, float all_positions[][2], float *fx, float *fy, float m);
int n_body(int n, int r);

int main(int argc, char **argv)
{
   int NUM_PROCESSORS;
   int rank;
  
   if (argc >= 2)
      NUM_BODIES = atoi(argv[1]);
   if (argc == 3)
      NUM_TIME_STEPS = atoi(argv[2]);

   MPI_Init(&argc, &argv);
   MPI_Comm_size(WORLD, &NUM_PROCESSORS);
   MPI_Comm_rank(WORLD, &rank);

   n_body(NUM_PROCESSORS, rank);
   
   MPI_Finalize();
   
   while (1) // so mpe window stays open
    usleep(1000000);
   
   return 0;
}

void printMatrix(float particles[][2])
{
   int i; int j;
   for (i=0; i<NUM_BODIES; i++)
   {
       printf("*#%d <%.2f,%.2f>\n",i, particles[i][0], particles[i][1]);
   }
}

int n_body(int NUM_PROCESSORS, int rank)
{
   int i; int t; int j; int k;
   float fx; float fy;
   float m = 4*pow(10,14);  // mass of objects
   int pos;
   int temp_pos;
   int left_rank, right_rank;
   MPI_Status status;
   
   MPE_XGraph handle;
   MPE_Open_graphics(&handle, WORLD, NULL, 0, 0, WINDOW_SIZE, WINDOW_SIZE, 0);

   left_rank = (rank == 0) ? (NUM_PROCESSORS-1) : (rank - 1);
   right_rank = (rank + 1) % NUM_PROCESSORS;
   
   int chunk_size = NUM_BODIES / NUM_PROCESSORS;  // particles per processor
   if (rank == 0)
   {
      printf("chunk size = bodies / proc = %d / %d = %d\n", NUM_BODIES, NUM_PROCESSORS, chunk_size);
   }
   srand(time(NULL) + rank);
      
   float all_positions[NUM_BODIES][2];
   float my_positions[chunk_size][2];   // each thread handles its own chunk    
   float recv_positions[chunk_size][2]; // buffer to store incoming positions
   float v[chunk_size][2];
   
   initialize_arrays(all_positions, v, chunk_size, rank);
   
   for (t=0; t<NUM_TIME_STEPS; t++)
   {
      // copy new calculations from all_positions array into my_positions so I can send them
      for (i=0; i<chunk_size; i++)
      {
         pos=rank*chunk_size+i;
         my_positions[i][0] = all_positions[pos][0];
         my_positions[i][1] = all_positions[pos][1];
      }
   
      // pass messages in a ring
      for (i=0; i<NUM_PROCESSORS-1; i++) 
      {
         if (rank % 2 == 1)
         {
			MPI_Send(&my_positions[0], chunk_size*2, MPI_FLOAT, right_rank, TAG, WORLD); // my_positions should have been sent, ok to modify now
			MPI_Recv(&my_positions[0], chunk_size*2, MPI_FLOAT, left_rank,  TAG, WORLD, &status);
			
         }
         else
         {
            MPI_Recv(&recv_positions[0], chunk_size*2, MPI_FLOAT, left_rank,  TAG, WORLD, &status);
            MPI_Send(&my_positions[0],   chunk_size*2, MPI_FLOAT, right_rank, TAG, WORLD);
           
		    for (j=0; j<chunk_size; j++)
            {
               my_positions[j][0] = recv_positions[j][0];
               my_positions[j][1] = recv_positions[j][1];
            }
         }
         
         for (k=0; k<chunk_size; k++)
         {
            // the following calculation puts the chunk in the right place in all_positions,
            //  ex. if the chunk is coming from rank#2, then put it in the 2nd chunk in the array
            pos = (NUM_PROCESSORS - i + left_rank) % NUM_PROCESSORS;
            all_positions[pos*chunk_size+k][0] = my_positions[k][0];
            all_positions[pos*chunk_size+k][1] = my_positions[k][1];
         }
      }
      
      for (i=0; i<chunk_size; i++)
      {
		 pos = rank*chunk_size+i;
         calculate_force(pos, all_positions, &fx, &fy, m); // compute force on pos's body
         
         // compute new velocity
         v[i][0] = v[i][0] + fx * dt / m;
         v[i][1] = v[i][1] + fy * dt / m; 
         
         // compute new position (leap-frog)
         all_positions[pos][0] = all_positions[pos][0] + v[i][0] * dt;
         all_positions[pos][1] = all_positions[pos][1] + v[i][1] * dt;
         
         // wrap around so bodies don't go off screen
         int div = all_positions[pos][0] / WINDOW_SIZE;
         if (div > 0)
         {
		    all_positions[pos][0] -= div * WINDOW_SIZE;
		 }
	     else if (div <= 0 && all_positions[pos][0] < 0)
	     {
	        all_positions[pos][0] += (abs(div)+1) * WINDOW_SIZE;
	     }
	           
	     div = all_positions[pos][1] / WINDOW_SIZE;
         if (div > 0)
         {
			all_positions[pos][1] -= div * WINDOW_SIZE;
	     }
	     else if (div <= 0 && all_positions[pos][1] < 0)
	     {
	        all_positions[pos][1] += (abs(div)+1) * WINDOW_SIZE;
	     }
	        
         //printf("rank#%d, particle#%d <%.6f, %.6f> v=%.6f,%.6f fx=%.6f, fy=%.6f\n", rank, pos, 
                    //all_positions[pos][0], all_positions[pos][1], v[i][0], v[i][1], fx, fy);
      }
      
      if (rank == 0) 
      {
         printf("****\nTime Step %d\n****\n", t);
         printMatrix(all_positions);
      }
      
      // MPE
      for (i=0; i<chunk_size; i++)
      {
         int num_colors;
         MPE_Num_colors(handle, &num_colors);
         int color = rank+i+1;
         if (color == 3) color=8; // yellow is hard to see
         MPE_Draw_circle(handle, all_positions[rank*chunk_size+i][0], all_positions[rank*chunk_size+i][1], CIRCLE_RAD, color%num_colors);
      }
      // wait until all processes are done calculations before updating screen

      MPE_Update(handle);
      MPI_Barrier(WORLD);

      usleep(80000);
      
      for (i=0; i<chunk_size; i++)
      {
         MPE_Draw_circle(handle, all_positions[rank*chunk_size+i][0], all_positions[rank*chunk_size+i][1], CIRCLE_RAD, MPE_WHITE);
      }
   }
   return 0;
}

int calculate_force(int particle_pos, float all_positions[][2], float *fx, float *fy, float m)
{
   float G = 6.67*pow(10,-11);
   float xdiff;
   float ydiff;
   float rSq;
   float F = 0;
   float fx_next = 0;
   float fy_next = 0;
   int i;

   for (i=0; i<NUM_BODIES; i++)
   {
      if (particle_pos != i)  // dont take bodies own body into account
      {
	     xdiff = all_positions[i][0] - all_positions[particle_pos][0];
		 ydiff = all_positions[i][1] - all_positions[particle_pos][1];
     
		 rSq = xdiff*xdiff + ydiff*ydiff;

		 if (rSq != 0)
		 {
		    F = (G * m * m) / rSq;  //otherwise F=0
		    
		    // if they are really close, forces get really big, so ignore
		    //  I'm not sure if this is the right thing to go but it makes
		    //  velocities not get out of control
		    if (rSq < 10)
		    { 
			   F = 0;
		    }
		    
		    // calculate x and y components of the force 
	        if (xdiff != 0) 
	        {
		       fx_next += F * xdiff / sqrt(rSq);
		    }
	  
	        if (ydiff != 0)
	        {
		       fy_next += F * ydiff / sqrt(rSq);
		    }
		 }
      }
   }

   *fx = fx_next;
   *fy = fy_next;

   return 0;
}

void initialize_arrays(float all_positions[][2],float v[][2], int chunk_size, int rank)
{
   int i;
   int pos;
   
   for (i=0; i<chunk_size; i++)
   {
      // initialize positions of my chunk, and keep them in the window size
      pos = (rank*chunk_size)+i;
      all_positions[pos][0] =   rand() % WINDOW_SIZE; 
      all_positions[pos][1] =  rand() % WINDOW_SIZE;
      
      // initialize velocities to 0
      v[i][0] = 0;
      v[i][1] = 0;
  
      //printf("pos# %d rank# %d <%.6f,%.6f> v=%.6f,%.6f\n", i, rank,
		//	 all_positions[pos][0], all_positions[pos][1], v[i][0], v[i][1]);
   }
}
