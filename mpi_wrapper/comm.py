from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        bytes_transferred = src_array.itemsize * src_array.size
        
        if rank == 0:
            # Root process: accumulate the reduction result
            np.copyto(dest_array, src_array)
            temp_buffer = np.empty_like(src_array)
            for i in range(1, size):
                self.comm.Recv(temp_buffer, source=i)
                # Apply reduction
                if op == MPI.SUM:
                    np.add(dest_array, temp_buffer, out=dest_array)
                elif op == MPI.MIN:
                    np.minimum(dest_array, temp_buffer, out=dest_array)
                elif op == MPI.MAX:
                    np.maximum(dest_array, temp_buffer, out=dest_array)
                else:
                    raise NotImplementedError("Only MPI.SUM, MPI.MIN, and MPI.MAX are supported.")

            # Broadcast the final reduced result
            for i in range(1, size):
                self.comm.Send(dest_array, dest=i)
            # Update total bytes transferred
            self.total_bytes_transferred += 2 * bytes_transferred * (size - 1)
        else:
            # Non-root processes: send data to root, then receive the final result
            self.comm.Send(src_array, dest=0)
            self.comm.Recv(dest_array, source=0)
            # Update total bytes transferred
            self.total_bytes_transferred += 2 * bytes_transferred

    # Non blocking Alltoall
    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        segment_size = src_array.size // size

        local_segment = rank * segment_size
        dest_array[local_segment:local_segment + segment_size] = src_array[local_segment:local_segment + segment_size]
        
        requests = []
        recv_buffers = {}

        for i in range(size):
            if i != rank:
                recv_buf = np.empty(segment_size, dtype=src_array.dtype)
                recv_buffers[i] = recv_buf
                req = self.comm.Irecv(recv_buf, source=i)
                requests.append(req)

        for i in range(size):
            if i != rank:
                send_start = i * segment_size
                send_buf = src_array[send_start:send_start+segment_size]
                req = self.comm.Isend(send_buf, dest=i)
                requests.append(req)

        MPI.Request.Waitall(requests)

        for i in range(size):
            if i != rank:
                dest_start = i * segment_size
                dest_array[dest_start:dest_start + segment_size] = recv_buffers[i]

        # Update total bytes transferred
        bytes_transferred = src_array.itemsize * segment_size # data for a single chunk
        self.total_bytes_transferred += 2 * bytes_transferred * (size - 1) # send and receive of (size - 1) chunks by each rank

    # Pairwise Alltoall
    def myAlltoall2(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        chunk_size = src_array.size // size  # Number of elements each process sends/receives
        recv_buffer = np.empty(chunk_size, dtype=dest_array.dtype)

        for i in range(size):
            start = i * chunk_size
            end = (i + 1) * chunk_size

            if i == rank:
                # Copy directly for local segment
                np.copyto(dest_array[start : end], src_array[start : end])
                # dest_array[start : end] = src_array[start : end]
            else:
                # Exchange with other processes
                self.comm.Sendrecv(src_array[start : end], dest=i, sendtag=rank,
                                recvbuf=recv_buffer, source=i, recvtag=i)
                np.copyto(dest_array[start : end], recv_buffer)
                # dest_array[start : end] = recv_buffer

        # Update total bytes transferred
        bytes_transferred = recv_buffer.itemsize * recv_buffer.size # data for a single chunk
        self.total_bytes_transferred += 2 * bytes_transferred * (size - 1) # send and receive of (size - 1) chunks by each rank



