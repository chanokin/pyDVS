import numpy
import pylab
import cv2
import spynnaker.pyNN as sim
import time

from pydvs.external_dvs_emulator_device_wrapper import DvsEmulatorDevice
from pydvs.external_dvs_emulator_device import ExternalDvsEmulatorDevice

from pydvs.virtual_cam import VirtualCam



def main():
    minutes = 0
    seconds = 30
    milliseconds = 0
    run_time = minutes*60*1000 + seconds*1000 + milliseconds

    weight_to_spike = 4.

    model = sim.IF_curr_exp
    cell_params = {'cm'        : 0.25, # nF
                    'i_offset'  : 0.0,
                    'tau_m'     : 10.0,
                    'tau_refrac': 2.0,
                    'tau_syn_E' : 2.5,
                    'tau_syn_I' : 2.5,
                    'v_reset'   : -70.0,
                    'v_rest'    : -65.0,
                    'v_thresh'  : -55.4
                    }
    # Available resolutions
    # 16, 32, 64, 128
    mode = ExternalDvsEmulatorDevice.MODE_64
    cam_res = int(mode)
    cam_fps = 90
    frames_per_saccade = cam_fps/3 - 1
    polarity = ExternalDvsEmulatorDevice.MERGED_POLARITY
    output_type = ExternalDvsEmulatorDevice.OUTPUT_TIME
    history_weight = 1.0
    behaviour = VirtualCam.BEHAVE_ATTENTION
    vcam = VirtualCam("./mnist", behaviour=behaviour, fps=cam_fps, 
                      resolution=cam_res, frames_per_saccade=frames_per_saccade)
                      
    cam_params = {'mode': mode,
                  'polarity': polarity,
                  'threshold': 12,
                  'adaptive_threshold': False,
                  'fps': cam_fps,
                  'inhibition': False,
                  'output_type': output_type,
                  'save_spikes': "./spikes_from_cam.pickle",
                  'history_weight': history_weight,
                  #'device_id': 0, # for an OpenCV webcam device
                  #'device_id': 'path/to/video/file', # to encode pre-recorded video
                  'device_id': vcam,
                 }
    if polarity == ExternalDvsEmulatorDevice.MERGED_POLARITY:
        num_neurons = 2*(cam_res**2)
    else:
        num_neurons = cam_res**2
      
    sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

    target = sim.Population(num_neurons, model, cell_params)

    stimulation = sim.Population(num_neurons, DvsEmulatorDevice, cam_params,
                                 label="Webcam population")

    connector = sim.OneToOneConnector(weights=weight_to_spike)

    projection = sim.Projection(stimulation, target, connector)

    target.record()
        
    sim.run(run_time)

    
    spikes = target.getSpikes(compatible_output=True)

    sim.end()
    #stimulation._vertex.stop()
    
    
    print ("Raster plot of the spikes that Spinnaker echoed back")
    fig = pylab.figure()
    
    spike_times = [spike_time for (neuron_id, spike_time) in spikes]
    spike_ids   = [neuron_id  for (neuron_id, spike_time) in spikes]
    
    pylab.plot(spike_times, spike_ids, ".", markerfacecolor="None",
               markeredgecolor="Blue", markersize=3)
    
    pylab.show()





#######################################################################
#######################################################################
#######################################################################


main()

#######################################################################
