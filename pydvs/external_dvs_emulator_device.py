import logging

spinn_version = "master"
try:
  from spinn_front_end_common.abstract_models.\
      abstract_provides_outgoing_edge_constraints import \
      AbstractProvidesOutgoingEdgeConstraints as AbstractProvidesOutgoingConstraints
  spinn_version = "2015.005"
except:
  from spinn_front_end_common.abstract_models.\
      abstract_provides_outgoing_partition_constraints import \
      AbstractProvidesOutgoingPartitionConstraints as AbstractProvidesOutgoingConstraints

from spinn_front_end_common.utility_models.reverse_ip_tag_multi_cast_source\
    import ReverseIpTagMultiCastSource
from pacman.model.constraints.key_allocator_constraints\
    .key_allocator_contiguous_range_constraint \
    import KeyAllocatorContiguousRangeContraint

from spynnaker_external_devices_plugin.pyNN.connections\
    .spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection

from spinnman.messages.eieio.command_messages.database_confirmation import \
    DatabaseConfirmation

import spynnaker_external_devices_plugin.pyNN as ExternalDevices

from spynnaker.pyNN import exceptions

from virtual_cam import VirtualCam

import cv2

import numpy as np
from numpy import uint8, int16
from multiprocessing import Process, Queue
import time
import pickle

import pyximport; pyximport.install()
import generate_spikes as gs

logger = logging.getLogger(__name__)


class ExternalDvsEmulatorDevice(ReverseIpTagMultiCastSource,
                    AbstractProvidesOutgoingConstraints,
                    ):

    MODE_128 = "128"
    MODE_64 = "64"
    MODE_32 = "32"
    MODE_16 = "16"

    UP_POLARITY = "UP"
    DOWN_POLARITY = "DOWN"
    MERGED_POLARITY = "MERGED"
    RECTIFIED_POLARITY = "RECTIFIED_POLARITY"
    POLARITY_DICT = {UP_POLARITY: uint8(0),
                     DOWN_POLARITY: uint8(1),
                     MERGED_POLARITY: uint8(2),
                     RECTIFIED_POLARITY: uint8(3),
                     0: UP_POLARITY,
                     1: DOWN_POLARITY,
                     2: MERGED_POLARITY,
                     3: RECTIFIED_POLARITY}

    OUTPUT_RATE = "RATE"
    OUTPUT_TIME = "TIME"
    OUTPUT_TIME_BIN = "TIME_BIN"
    OUTPUT_TIME_BIN_THR = "TIME_BIN_THR"

    def __init__(self, n_neurons, machine_time_step, timescale_factor,
                 database_socket,
                 label, port=12345, virtual_key=None,
                 spikes_per_second=0, ring_buffer_sigma=None,
                 device_id=0, fps=60, mode="128", scale_img=True, polarity="MERGED",
                 inhibition = False, inh_area_width = 2,
                 threshold=12, adaptive_threshold = False,
                 min_threshold=6, max_threshold=168,
                 threshold_delta_down = 2, threshold_delta_up = 12,
                 output_type="TIME", num_bits_per_spike=4,
                 history_weight=0.99, save_spikes=None, run_time_ms=None,
                 local_port=19876):
        """
        :param device_id: int for webcam modes, or string for video file
        :param mode: The retina "mode"
        :param retina_key: The value of the top 16-bits of the key
        :param polarity: The "polarity" of the retina data
        :param machine_time_step: The time step of the simulation
        :param timescale_factor: The timescale factor of the simulation
        :param label: The label for the population
        :param n_neurons: The number of neurons in the population

        """
        fixed_n_neurons = n_neurons

        if mode == ExternalDvsEmulatorDevice.MODE_128 or \
           mode == ExternalDvsEmulatorDevice.MODE_64  or \
           mode == ExternalDvsEmulatorDevice.MODE_32  or \
           mode == ExternalDvsEmulatorDevice.MODE_16:
            self._out_res = int(mode)
            self._res_2x = self._out_res*2

        else:
            raise exceptions.SpynnakerException("the model does not "
                                                "recongise this mode")

        if (polarity == ExternalDvsEmulatorDevice.UP_POLARITY or
            polarity == ExternalDvsEmulatorDevice.DOWN_POLARITY or
            polarity == ExternalDvsEmulatorDevice.RECTIFIED_POLARITY):
            fixed_n_neurons = self._out_res**2
        else:
            fixed_n_neurons = 2*(self._out_res**2)

        if fixed_n_neurons != n_neurons and n_neurons is not None:
            logger.warn("The specified number of neurons for the DVS emulator"
                        " device has been ignored {} will be used instead"
                        .format(fixed_n_neurons))

        self._video_source = None
        self._device_id = device_id
        self._is_virtual_cam = False
        self._polarity = polarity
        self._polarity_n = ExternalDvsEmulatorDevice.POLARITY_DICT[polarity]
        self._global_max = int16(0)
        self._output_type = output_type

        self._raw_frame = None
        self._gray_frame = None
        self._tmp_frame = None

        self._ref_frame = 128*np.ones((self._out_res, self._out_res), dtype=int16)

        self._curr_frame = np.zeros((self._out_res, self._out_res), dtype=int16)

        self._spikes_frame = np.zeros((self._out_res, self._out_res, 3), dtype=uint8)

        self._diff = np.zeros((self._out_res, self._out_res), dtype=int16)

        self._abs_diff = np.zeros((self._out_res, self._out_res), dtype=int16)

        self._spikes = np.zeros((self._out_res, self._out_res), dtype=int16)

        self._adaptive_threshold = adaptive_threshold
        self._thresh_matrix = None
        if adaptive_threshold:
          self._thresh_matrix = np.zeros((self._out_res, self._out_res),
                                             dtype=int16)

        self._threshold_delta_down = int16(threshold_delta_down)
        self._threshold_delta_up = int16(threshold_delta_up)
        self._max_threshold = int16(max_threshold)
        self._min_threshold = int16(min_threshold)

        self._up_spikes = None
        self._down_spikes = None
        self._spikes_lists = None

        self._threshold = int16(threshold)

        self._data_shift = uint8(np.log2(self._out_res))
        self._up_down_shift = uint8(2*self._data_shift)
        self._data_mask = uint8(self._out_res - 1)


        if self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME_BIN:
            self._num_bins = 8 #8-bit images don't need more
        elif self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME_BIN_THR:
            self._num_bins = 6 #should be enough?
        else:
            self._num_bins = int(1000./fps)

        self._num_bits_per_spike = min(num_bits_per_spike, self._num_bins)

        if self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME_BIN or \
           self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME_BIN_THR:
            self._log2_table = gs.generate_log2_table(self._num_bits_per_spike, self._num_bins)
        else:
            self._log2_table = gs.generate_log2_table(self._num_bits_per_spike, 8) #stupid hack, compatibility issues


        self._scale_img = scale_img
        self._img_height = 0
        self._img_height_crop_u = 0
        self._img_height_crop_b = 0
        self._img_width = 0
        self._img_width_crop_l = 0
        self._img_width_crop_r = 0
        self._img_ratio = 0.
        self._img_scaled_width = 0
        self._scaled_width = 0
        self._fps = fps
        self._max_time_ms = 0
        self._time_per_frame = 0.

        self._time_per_spike_pack_ms = 0

        self._get_sizes = True
        self._scale_changed = False

        self._running = True

        self._label = label
        self._n_neurons = fixed_n_neurons
        self._local_port = local_port

        self._inh_area_width = inh_area_width
        self._inhibition = inhibition
        self._inh_coords = gs.generate_inh_coords(self._out_res,
                                               self._out_res,
                                               inh_area_width)

        self._history_weight = history_weight

        self._run_time_ms = run_time_ms
        ################################################################

        if spinn_version == "2015.005":
            ReverseIpTagMultiCastSource.__init__(self,
                n_neurons=self._n_neurons,
                machine_time_step=machine_time_step,
                timescale_factor=timescale_factor,
                port=self._local_port,
                label=self._label,
                virtual_key=virtual_key)
        else:
            ReverseIpTagMultiCastSource.__init__(self,
                n_keys=self._n_neurons,
                machine_time_step=machine_time_step,
                timescale_factor=timescale_factor,
                label=self._label,
                receive_port=self._local_port,
                virtual_key=virtual_key)

        AbstractProvidesOutgoingConstraints.__init__(self)

        print("number of neurons for webcam = %d"%self._n_neurons)

        self._live_conn = SpynnakerLiveSpikesConnection(send_labels = [self._label, ],
                                                        local_port = self._local_port)
        def init(label, n_neurons, run_time_ms, machine_timestep_ms):
            print("Sending %d neuron sources from %s"%(n_neurons, label))

        self._live_conn.add_init_callback(self._label, init)
        self._live_conn.add_start_callback(self._label, self.run)
        self._sender = None

        self._save_spikes = save_spikes


    def get_outgoing_edge_constraints(self, partitioned_edge, graph_mapper):
        constraints = ReverseIpTagMultiCastSource\
            .get_outgoing_edge_constraints(
                self, partitioned_edge, graph_mapper)
        constraints.append(KeyAllocatorContiguousRangeContraint())
        return constraints


    def get_number_of_mallocs_used_by_dsg(self, vertex_slice, in_edges):
        mallocs = \
            ReverseIpTagMultiCastSource.get_number_of_mallocs_used_by_dsg(
                self, vertex_slice, in_edges)
        if config.getboolean("SpecExecution", "specExecOnHost"):
            return 1
        else:
            return mallocs


    def __del__(self):

        self._running = False


    def stop(self):
        self.__del__()


    def run(self, label, sender):

        self._label = label
        self._sender = sender


        if self._run_time_ms is None:
            max_run_time_s = self.no_machine_time_steps/float(self.machine_time_step) - 0.5
        else:
            max_run_time_s = self._run_time_ms/1000.


        self.acquire_device()

        spike_queue = Queue()
        spike_emmision_proc = Process(target=self.send_spikes, args=(spike_queue,))
        spike_emmision_proc.start()

        img_queue = Queue()
        #~ spike_gen_proc = Process(target=self.process_frame, args=(img_queue,))
        spike_gen_proc = Process(target=self.process_frame, args=(img_queue, spike_queue))
        spike_gen_proc.start()

        grab_times = []
        start_time = 0.
        app_start_time = time.time()
        app_curr_time = time.time()
        first_frame = True
        frame_time = 0.
        while self._running:

            start_time = time.time()
            valid_frame = self.grab_frame()
            grab_times.append(time.time() - start_time)

            if not valid_frame:
                self._running = False

            # send the minimum difference value once to synchronize time-based
            # decoding on the receiver end
            if self._output_type != ExternalDvsEmulatorDevice.OUTPUT_RATE and \
               first_frame == True:

                first_frame = False
                self._curr_frame[:] = 128 + self._threshold + 1 #half range of vals + thresh + 1


            img_queue.put(self._curr_frame)

            app_curr_time = time.time()
            if app_curr_time - app_start_time > max_run_time_s:
              self._running = False
            frame_time = time.time() - start_time
            if frame_time < self._time_per_frame:
              time.sleep(self._time_per_frame - frame_time)

        print("webcam runtime ", app_curr_time - app_start_time)
        img_queue.put(None)
        spike_gen_proc.join()

        spike_queue.put(None)
        spike_emmision_proc.join()

        if self._video_source is not None:
            self._video_source.release()
            cv2.destroyAllWindows()


    def process_frame(self, img_queue, spike_queue):

        label = self._label
        spikes_frame = self._spikes_frame
        
        cv2.namedWindow (label)
        cv2.startWindowThread()
        
        res2x = self._res_2x
        spike_list = []
        # gen_times = []
        # compose_times = []
        # transform_times = []
        # ref_up_times = []
        lists = None
        while True:
            image = img_queue.get()

            if image is None or not self._running:
                break

            # start_time = time.time()
            self.generate_spikes(image)
            # gen_times.append(time.time()-start_time)

            # start_time = time.time()
            self.update_reference()
            # ref_up_times.append(time.time()-start_time)

            # start_time = time.time()
            lists = self.transform_spikes()
            # transform_times.append(time.time() - start_time)

            spike_queue.put(lists)

            if self._save_spikes is not None:
                spike_list.append(lists)

            # start_time = time.time()
            self.compose_output_frame()
            # compose_times.append(time.time()-start_time)


            cv2.imshow( label, cv2.resize(spikes_frame, (res2x, res2x)) )

            if cv2.waitKey(1) & 0xFF == ord('q'):#\
              #or not sender.isAlive():
              self._running = False
              break

            #continue

        # print("gen times")
        # print(np.array(gen_times).mean())
        # print("update ref times")
        # print(np.array(ref_up_times).mean())
        # print("transform times")
        # print(np.array(transform_times).mean())
        # print("compose times")
        # print(np.array(compose_times).mean())

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        if self._save_spikes is not None:
            #print spike_list
            print("attempting to save spike_list")
            pickle.dump( spike_list, open(self._save_spikes, "wb") )


    def send_spikes(self, spike_queue):
        sender = self._sender
        while True:
            spikes = spike_queue.get()

            if spikes is None or not self._running:
                break

            self.emit_spikes(sender, spikes)




    def acquire_device(self):
        if isinstance(self._device_id, VirtualCam):
            self._video_source = self._device_id
            self._is_virtual_cam = True
        else:
            self._video_source = cv2.VideoCapture(self._device_id)

        if not self._video_source.isOpened():
            self._video_source.open()

        try:
            self._video_source.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self._video_source.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        except:
            pass

        try:
            self._video_source.set(cv2.CAP_PROP_FPS, self._fps)
        except:
            self._fps = self._video_source.get(cv2.CAP_PROP_FPS)

        self._max_time_ms = int16((1./self._fps)*1000)
        self._time_per_frame = 1./self._fps

        self._time_per_spike_pack_ms = self.calculate_time_per_pack()




    def grab_frame(self):
        #~ start_time = time.time()
        if self._is_virtual_cam:
            valid_frame, self._curr_frame[:] = self._video_source.read(self._ref_frame)
            return True

        else:
            if self._raw_frame is None or self._scale_changed:
                valid_frame, self._raw_frame = self._video_source.read()
            else:
                valid_frame, self._raw_frame[:] = self._video_source.read()

            #~ end_time = time.time()
            #~ print("Time to capture frame = ", end_time - start_time)

            if not valid_frame:
              return False

            #~ start_time = time.time()
            if self._gray_frame is None or self._scale_changed:
                self._gray_frame = cv2.convertColor(self._raw_frame, cv2.COLOR_BGR2GRAY).astype(int16)
            else:
                self._gray_frame[:] = cv2.convertColor(self._raw_frame, cv2.COLOR_BGR2GRAY)
            #~ end_time = time.time()
            #~ print("Time to convert to grayscale = ", end_time - start_time)

            #~ start_time = time.time()
            if self._get_sizes or self._scale_changed:
                self._get_sizes = False
                self._scale_changed = False
                self._img_height, self._img_width = self._gray_frame.shape

                self._img_ratio = float(self._img_width)/float(self._img_height)
                self._img_scaled_width = int(float(self._out_res)*self._img_ratio)

                if self._scale_img:
                    diff = self._img_scaled_width - self._out_res
                    self._img_width_crop_l = diff//2
                    self._img_width_crop_r = self._img_width_crop_l + self._out_res
                else:
                    diff = self._img_width - self._out_res
                    self._img_width_crop_l = diff//2
                    self._img_width_crop_r = self._img_width_crop_l + self._out_res
                    diff = self._img_height - self._out_res
                    self._img_height_crop_u = diff//2
                    self._img_height_crop_b = self._img_height_crop_u + self._out_res

                self._tmp_frame = np.zeros((self._out_res, self._img_scaled_width))


            #~ end_time = time.time()
            #~ print("Time to calculate sizes = ", end_time - start_time)

            #~ start_time = time.time()
            if self._scale_img:
                self._tmp_frame[:] = cv2.resize(self._gray_frame, (self._img_scaled_width, self._out_res),
                                             interpolation=cv2.INTER_NN)

                self._curr_frame[:] = self._tmp_frame[:, self._img_width_crop_l: self._img_width_crop_r]
            else:
                self._curr_frame[:] = self._gray_frame[self._img_height_crop_u: self._img_height_crop_b,
                                                    self._img_width_crop_l:  self._img_width_crop_r]
            #~ end_time = time.time()
            #~ print("Time to scale frame = ", end_time - start_time)


        return True






    def emit_spikes(self, sender, lists):
        lbl     = self._label
        max_time_s = self._time_per_spike_pack_ms/1000.
        #~ lists   = self._spikes_lists
        send_spikes = sender.send_spikes

        #from generate_spikes.pyx (cython)
        if lists is not None:
          for spike_pack in lists:
            start_time = time.time()
            send_spikes(lbl, spike_pack, send_full_keys=False)
            elapsed_time = time.time() - start_time
            if elapsed_time < max_time_s:
              time.sleep(max_time_s - elapsed_time)



    def generate_spikes(self, image):
        self._curr_frame = image
        curr_frame = self._curr_frame

        #~ curr_frame = image
        ref_frame = self._ref_frame
        diff = self._diff
        abs_diff = self._abs_diff
        spikes = self._spikes
        img_w = self._out_res
        inh_w = self._inh_area_width
        inhibition = self._inhibition
        inh_coords = self._inh_coords
        threshold = self._threshold

        if self._adaptive_threshold:
            thresh_mat = self._thresh_matrix


            #all from generate_spikes.pyx (cython)
            diff[:], abs_diff[:], spikes[:] = gs.thresholded_difference_adpt(curr_frame, ref_frame,
                                                                          thresh_mat)
        else:
            #all from generate_spikes.pyx (cython)
            diff[:], abs_diff[:], spikes[:] = gs.thresholded_difference(curr_frame, ref_frame,
                                                                     threshold)

        if inhibition:
            spikes[:] = gs.local_inhibition(spikes, abs_diff, inh_coords, img_w, img_w, inh_w)




    def update_reference(self):
        abs_diff = self._abs_diff
        spikes = self._spikes
        ref_frame = self._ref_frame
        min_thresh = self._min_threshold
        max_thresh = self._max_threshold
        threshold  = self._threshold
        max_time_ms = self._max_time_ms
        history_weight = self._history_weight
        num_bits = self._num_bits_per_spike
        log2_table = self._log2_table[num_bits-1] #no values for 0-bit encoding

        if self._adaptive_threshold:
            thresh_mat = self._thresh_matrix
            thresh_delta_down = self._threshold_delta_down
            thresh_delta_up   = self._threshold_delta_up
            ref_frame[:], thresh_mat[:] = gs.update_reference_rate_adpt(abs_diff, spikes,
                                                                     ref_frame, thresh_mat,
                                                                     min_thresh, max_thresh,
                                                                     thresh_delta_down,
                                                                     thresh_delta_up,
                                                                     max_time_ms,
                                                                     history_weight)

        else:
            if self._output_type == ExternalDvsEmulatorDevice.OUTPUT_RATE:

                ref_frame[:] = gs.update_reference_rate(abs_diff, spikes, ref_frame,
                                                     threshold, max_time_ms, history_weight)

            elif self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME:

                ref_frame[:] = gs.update_reference_time_thresh(abs_diff, spikes, ref_frame,
                                                            threshold, max_time_ms,
                                                            history_weight)

            elif self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME_BIN:

                ref_frame[:] = gs.update_reference_time_binary_raw(abs_diff, spikes, ref_frame,
                                                                threshold, max_time_ms,
                                                                num_bits,
                                                                history_weight,
                                                                log2_table)

            elif self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME_BIN_THR:

                ref_frame[:] = gs.update_reference_time_binary_thresh(abs_diff, spikes, ref_frame,
                                                                   threshold, max_time_ms,
                                                                   num_bits,
                                                                   history_weight,
                                                                   log2_table)

    def transform_spikes(self):
        data_shift = self._data_shift
        up_down_shift = self._up_down_shift
        data_mask = self._data_mask
        polarity = self._polarity_n
        spikes = self._spikes
        threshold = self._threshold
        max_thresh = self._max_threshold
        #~ lists = self._spikes_lists
        max_time_ms = self._max_time_ms
        abs_diff = self._abs_diff
        num_bins = self._num_bins
        num_bits = self._num_bits_per_spike
        log2_table = self._log2_table[num_bits - 1]

        dn_spks, up_spks, g_max = gs.split_spikes(spikes, abs_diff, polarity)
        lists = None
        #from generate_spikes.pyx (cython)
        if self._output_type == ExternalDvsEmulatorDevice.OUTPUT_RATE:
          lists = gs.make_spike_lists_rate(up_spks, dn_spks,
                                        g_max, threshold,
                                        up_down_shift, data_shift, data_mask,
                                        max_time_ms)

        elif self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME:
          lists = gs.make_spike_lists_time(up_spks, dn_spks,
                                        g_max,
                                        up_down_shift, data_shift, data_mask,
                                        num_bins,
                                        max_time_ms,
                                        threshold, max_thresh)

        elif self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME_BIN:
          lists = gs.make_spike_lists_time_bin(up_spks, dn_spks,
                                            g_max,
                                            up_down_shift, data_shift, data_mask,
                                            max_time_ms,
                                            threshold, max_thresh,
                                            num_bins,
                                            log2_table)

        elif self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME_BIN_THR:
          lists = gs.make_spike_lists_time_bin_thr(up_spks, dn_spks,
                                                  g_max,
                                                  up_down_shift, data_shift, data_mask,
                                                  max_time_ms,
                                                  threshold, max_thresh,
                                                  num_bins,
                                                  log2_table)
        return lists


    def compose_output_frame(self):
        curr_frame = self._curr_frame
        spikes_frame = self._spikes_frame
        spikes = self._spikes
        width = self._out_res
        height = self._out_res
        polarity = self._polarity_n

        #from generate_spikes.pyx (cython)
        spikes_frame[:] = gs.render_frame(spikes=spikes, curr_frame=curr_frame,
                                          width=width, height=height,
                                          polarity=polarity)


    def calculate_time_per_pack(self):
        time_per_pack = 0
        if self._output_type == ExternalDvsEmulatorDevice.OUTPUT_RATE:
            time_per_pack = 1

        elif self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME:
            time_per_pack = (self._max_time_ms)// \
                            (min(self._max_time_ms,
                                 self._max_threshold//self._min_threshold + 1))

        elif self._output_type == ExternalDvsEmulatorDevice.OUTPUT_TIME_BIN:
            time_per_pack = (self._max_time_ms)//(8) #raw difference value could be 8-bit
        else:
            time_per_pack = (self._max_time_ms)//(self._num_bits_per_spike + 1)

        return time_per_pack
