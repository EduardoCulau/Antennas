from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

class Antenna(ABC):
    @abstractmethod
    def gain(self, **Kwargs):
        pass

class DipoleAntenna(Antenna):
    def gain(self, **Kwargs):
        return tbd

class PhasedArrayAntenna(Antenna):
    # Number of elements/antennas in the Array
    N = None
    
    def __init__(self, N=1):
        self.N = N
    
    def gain(self, **Kwargs):
        # Parallel Computing
        n_jobs = Kwargs.get('n_jobs', 1)
        verbose = Kwargs.get('verbose', 1)
        
        # Simulation Parameters
        N = Kwargs.get('N', self.N)
        k = Kwargs.get('k', 0)
        d = Kwargs.get('d', 0)
        angles = Kwargs.get('angles', 0)
        
        # Update Class Variables
        self.N = N
        
        # Divide work into chunks
        chunk_size = N // n_jobs
        chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_jobs)]
        chunks[-1] = (chunks[-1][0], N)  # Ensure the last chunk goes up to N
        
        # Assign thread numbers to each chunk
        thread_numbers = list(range(n_jobs))
        
        # Log the resume of the simlation
        if verbose:
            print("This Simulation will run now with %d threads and each of them will process a %d elements." % (n_jobs, chunk_size))
        
        # Use joblib.Parallel to compute array factors in parallel
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.__compute_chunk)(thread_num, start, end, angles, k, d) for (start, end), thread_num in zip(chunks, thread_numbers) 
        )
        
        # All Threads Complete
        if verbose:
            print('All Threads Completed')
        
        # Combine results
        array_factor = np.sum(results, axis=0)

        # Calculate the gain pattern (magnitude of array factor squared)
        gain = np.abs(array_factor) ** 2

        # Normalize the gain
        gain = 10 * np.log10(gain)
        
        return gain
    
    def __compute_chunk(self, thread_num, start, end, angles, k, d):
        # Log to Starting Computation
        elements = (end - start)
        print(f"Thread ID: {thread_num} start processing {elements} elements from {start} to {end}.")
        
        # Pre-Compute the Angles
        sin_angles = np.sin(angles)
        
        # Compute
        chunk_factor = np.zeros_like(angles, dtype=complex)
        for n in range(start, end):
            chunk_factor += np.exp(1j * k * d * n * np.sin(angles))
            
            #Log Completion
            percent_complete  = ((n - start+1) / elements) * 100
            if percent_complete % 1 == 0:
                print(f"Thread {thread_num} progress: {percent_complete:.0f}%")
        return chunk_factor
    
    def plot(self, **Kwargs):
        # Create polar plot
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.plot(self.angles, self.gain)
        ax.set_title("Polar Plot of a Phased Array Antenna Gain with %d Antennas" % N, va='bottom')

class SimulateAntenna:
    # Variables
    angles = None;
    gain = None;
    
    def __init__(self, antenna: Antenna, **Kwargs):
        self.angles = Kwargs.get('angles', None)
        self._antenna = antenna

    def set_antenna(self, antenna: Antenna):
        self._antenna = antenna

    def gain(self, **Kwargs):
        self.angles = Kwargs.get('angles', None)
        self.gain = self._antenna.gain(**Kwargs)
        return self._antenna.gain(**Kwargs)
        
    def plot(self, **Kwargs):
        # Create polar plot
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.plot(self.angles, self.gain)
        ax.set_title("Polar Plot of a Phased Array Antenna Gain with %d Antennas" % self._antenna.N, va='bottom')
    
    def show(self):
        # Display the plot
        plt.show()