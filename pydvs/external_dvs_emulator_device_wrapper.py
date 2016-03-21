from spinnman.messages.eieio.eieio_type import EIEIOType

from spynnaker_external_devices_plugin.pyNN import model_binaries
from spynnaker_external_devices_plugin.pyNN.\
    spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager


from spynnaker.pyNN.utilities import conf
from spynnaker.pyNN import IF_curr_exp
from spynnaker.pyNN.spinnaker import executable_finder

from external_dvs_emulator_device import ExternalDvsEmulatorDevice

from spinn_front_end_common.utilities.notification_protocol.socket_address \
    import SocketAddress

import os

executable_finder.add_path(os.path.dirname(model_binaries.__file__))
spynnaker_external_devices = SpynnakerExternalDevicePluginManager()



def DvsEmulatorDevice(n_neurons, machine_time_step, timescale_factor,
                      label, port=12345, virtual_key=None, 
                      spikes_per_second=0, ring_buffer_sigma=None,
                      device_id=0, fps=30, mode="128", scale_img=True, polarity="MERGED",
                      inhibition = False, inh_area_width = 2,
                      threshold=12, adaptive_threshold = False,
                      min_threshold= 6, max_threshold=168,
                      threshold_delta_down = 2, threshold_delta_up = 12,
                      output_type="TIME", num_bits_per_spike=4, 
                      history_weight=0.99, save_spikes=None,
                      local_port=19999,
                      database_notify_host=None, database_notify_port_num=None,
                      database_ack_port_num=None):

    """
    supports adding a spike injector to the applciation graph.
    :param n_neurons: the number of neurons the spike injector will emulate
    :type n_neurons: int
    :param machine_time_step: the time period in ms for each timer callback
    :type machine_time_step: int
    :param timescale_factor: the amount of scaling needed of the machine time
    step (basically a slow down function)
    :type timescale_factor: int
    :param spikes_per_second: the expected number of spikes per second
    :type spikes_per_second: int
    :param ring_buffer_sigma: the number of standard divations from a
    calcuation on how much safety in percision vs overflowing
     (this is deduced from the front end)
    :type ring_buffer_sigma: int
    :param label: the label given to the population
    :type label: str
    :param port: the port number used to listen for injections of spikes
    :type port: int
    :param virtual_key: the virtual key used in the routing system
    :type virtual_key: int
    :param database_notify_host: the hostnmae for the device which is listening
    to the database notifcation.
    :type database_notify_host: str
    :param database_ack_port_num: the port number to which a external device
    will ack that they have finished reading the database and are ready for
    it to start execution
    :type database_ack_port_num: int
    :param database_notify_port_num: The port number to which a external device
    will recieve the database is ready command
    :type database_notify_port_num: int

    :return:
    """
    if database_notify_port_num is None:
        database_notify_port_num = conf.config.getint("Database",
                                                      "notify_port")
    if database_notify_host is None:
        database_notify_host = conf.config.get("Database", "notify_hostname")
    if database_ack_port_num is None:
        database_ack_port_num = conf.config.get("Database", "listen_port")
        if database_ack_port_num == "None":
            database_ack_port_num = None

    # build the database socket address used by the notifcation interface
    database_socket = SocketAddress(
        listen_port=database_ack_port_num,
        notify_host_name=database_notify_host,
        notify_port_no=database_notify_port_num)
    # update socket interface with new demands.
    spynnaker_external_devices.add_socket_address(database_socket)
    return ExternalDvsEmulatorDevice(
        n_neurons=n_neurons, machine_time_step=machine_time_step,
        timescale_factor=timescale_factor, 
        database_socket=database_socket,
        spikes_per_second=spikes_per_second, ring_buffer_sigma=ring_buffer_sigma, 
        label=label, port=port, virtual_key=virtual_key,
        device_id=device_id, fps=fps, mode=mode, scale_img=scale_img, polarity=polarity,
        inhibition=inhibition, inh_area_width = inh_area_width,
        threshold=threshold, adaptive_threshold=adaptive_threshold,
        min_threshold=min_threshold, max_threshold=max_threshold,
        threshold_delta_down=threshold_delta_down, threshold_delta_up=threshold_delta_up,
        output_type=output_type, num_bits_per_spike=num_bits_per_spike, 
        history_weight=history_weight, save_spikes=save_spikes,
        local_port=local_port)
        

