#use the information in sim_config.md to create a type and value checker for the initialization of the simulation

#nondict items, if value is not predefined, then use any to check for any value
version = {
    'type': str,
    'value': any
}
length_unit = {
    'type': str,
    'value': ['um']
}
space_unit = {
    'type': str,
    'value': ['pixel']
}
time_unit = {
    'type': str,
    'value': ['ms']
}
intensity_unit = {
    'type': str,
    'value': ['AUD']
}
diffusion_unit = {
    'type': str,
    'value': ['um^2/s']
}

#dict items, if value is not predefined, then use any to check for any value
#Cell_parameters
Cell_Parameters = {
    'type': dict,
    'value': any
}

#Track_parameters
Track_Parameters = {
    'type': dict,
    'value': any
}

Global_Parameters = {
    'type': dict,
    'value': any
}

Condensate_Parameters = {
    'type': dict,
    'value': any
}

#populate the values of the dictonaries TODO
