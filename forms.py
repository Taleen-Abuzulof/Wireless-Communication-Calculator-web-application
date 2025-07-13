# forms.py - Using Flask-WTF for consistent form handling
from flask_wtf import FlaskForm
from wtforms import  SelectField,  DecimalField, FloatField, IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange, ValidationError , InputRequired


def positive_check(form, field):
    if field.data <= 0:
        raise ValidationError('Value must be positive.')


class OfdmSystems(FlaskForm):
    bandwidth = FloatField(
        'Bandwidth (kHz)',
        validators=[
            DataRequired(message="Bandwidth is required"),
            NumberRange(min=1.0, max=1000000.0, message="Bandwidth must be between 1 and 1,000,000 kHz")
        ],
        render_kw={
            "placeholder": "e.g., 20000",
            "class": "form-control",
            "step": "0.1",
            "min": "1",
            "max": "1000000"
        }
    )
    
    subcarrier_spacing = FloatField(
        'Subcarrier Spacing (kHz)',
        validators=[
            DataRequired(message="Subcarrier spacing is required"),
            NumberRange(min=0.1, max=1000.0, message="Subcarrier spacing must be between 0.1 and 1000 kHz")
        ],
        render_kw={
            "placeholder": "e.g., 15",
            "class": "form-control",
            "step": "0.1",
            "min": "0.1",
            "max": "1000"
        }
    )
    
    ofdm_symbols = IntegerField(
        'Number of OFDM Symbols',
        validators=[
            DataRequired(message="Number of OFDM symbols is required"),
            NumberRange(min=1, max=1000, message="Number of symbols must be between 1 and 1000")
        ],
        render_kw={
            "placeholder": "e.g., 14",
            "class": "form-control",
            "min": "1",
            "max": "1000"
        }
    )
    
    modulation_type = SelectField(
        'Modulation Type',
        choices=[
            ('', 'Select Modulation'),
            ('2', 'BPSK (1 bit/symbol)'),
            ('4', 'QPSK (2 bits/symbol)'),
            ('16', '16-QAM (4 bits/symbol)'),
            ('64', '64-QAM (6 bits/symbol)'),
            ('256', '256-QAM (8 bits/symbol)'),
            ('1024', '1024-QAM (10 bits/symbol)')
        ],
        validators=[DataRequired(message="Please select a modulation type")],
        render_kw={
            "class": "form-control"
        }
    )
    
    # FIXED: More lenient validation for block duration
    block_duration = FloatField(
        'Block Duration (ms)',
        validators=[
            DataRequired(message="Block duration is required")
            # Removed the problematic NumberRange validator temporarily
        ],
        render_kw={
            "placeholder": "e.g., 1",
            "class": "form-control",
            "step": "0.1",
            "min": "0.1",
            "max": "1000"
        }
    )
    
    resource_blocks = IntegerField(
        'Number of Parallel Resource Blocks',
        validators=[
            DataRequired(message="Number of resource blocks is required"),
            NumberRange(min=1, max=10000, message="Number of resource blocks must be between 1 and 10,000")
        ],
        render_kw={
            "placeholder": "e.g., 100",
            "class": "form-control",
            "min": "1",
            "max": "10000"
        }
    )
    
    submit = SubmitField(
        'Calculate',
        render_kw={
            "class": "btn btn-calculate"
        }
    )
    
    # Custom validation with better error messages
    def validate_block_duration(self, field):
        """Custom validation for block duration"""
        if field.data is not None:
            try:
                value = float(field.data)
                if value <= 0:
                    raise ValidationError('Block duration must be greater than 0')
                elif value < 0.1:
                    raise ValidationError('Block duration must be at least 0.1 ms')
                elif value > 1000:
                    raise ValidationError('Block duration must be less than 1000 ms')
            except (TypeError, ValueError):
                raise ValidationError('Block duration must be a valid number')
    
    def validate_bandwidth(self, field):
        """Custom validation for bandwidth"""
        if field.data is not None and self.subcarrier_spacing.data is not None:
            try:
                bandwidth = float(field.data)
                subcarrier_spacing = float(self.subcarrier_spacing.data)
                
                if bandwidth <= 0:
                    raise ValidationError('Bandwidth must be greater than 0')
                elif subcarrier_spacing > bandwidth:
                    raise ValidationError('Bandwidth must be larger than subcarrier spacing')
            except (TypeError, ValueError):
                pass  # Let other validators handle type errors
    
    def validate_subcarrier_spacing(self, field):
        """Custom validation for subcarrier spacing"""
        if field.data is not None:
            try:
                value = float(field.data)
                if value <= 0:
                    raise ValidationError('Subcarrier spacing must be greater than 0')
            except (TypeError, ValueError):
                raise ValidationError('Subcarrier spacing must be a valid number')
    
    def validate_modulation_type(self, field):
        """Validate modulation type selection"""
        valid_modulations = ['2', '4', '16', '64', '256', '1024']
        if field.data and field.data not in valid_modulations:
            raise ValidationError('Please select a valid modulation type')
            
class CommunicationSystemForm(FlaskForm):
    bandwidth = DecimalField(
        'Analog Bandwidth (kHz)',
        validators=[DataRequired(), positive_check]
    )
    quantizer_bits = IntegerField(
        'Quantizer Bits',
        validators=[DataRequired(), positive_check]
    )
    compression_rate = DecimalField(
        'Source Encoder Compression Rate',
        validators=[DataRequired(), NumberRange(min=0, max=1)]
    )
    channel_encoder_rate = DecimalField(
        'Channel Encoder Rate',
        validators=[DataRequired(), NumberRange(min=0, max=1)]
    )
    voice_segment = DecimalField(
        'Voice Segment (ms)',
        validators=[DataRequired(), positive_check],
        render_kw={"placeholder": "Unit in ms"}
    )

    submit = SubmitField('Calculate')

# Validators
def positive_check(form, field):
    if field.data <= 0:
        raise ValidationError('Value must be positive.')

def non_negative_check(form, field):
    if field.data < 0:
        raise ValidationError('Value must be zero or positive.')

class LinkBudgetForm(FlaskForm):
    # Losses (must be positive)
    path_loss = DecimalField(
        'Path Loss (LP)',
        validators=[DataRequired(), positive_check]
    )
    path_loss_unit = SelectField(
        'Unit', choices=[('dB','dB'),('dBm','dBm'),('watt','Watt')],
        validators=[DataRequired()]
    )

    # Frequency (must be positive)
    frequency = DecimalField(
        'Frequency (MHz)',
        validators=[DataRequired(), positive_check]
    )

    # Antenna gains (can be zero or positive)
    transmitter_antenna_gain = DecimalField(
        'Transmitter Antenna Gain (Gt)',
        validators=[InputRequired(), non_negative_check]
    )
    transmitter_antenna_gain_unit = SelectField(
        'Unit', choices=[('dB','dB'),('dBm','dBm'),('watt','Watt')],
        validators=[DataRequired()]
    )

    receiver_antenna_gain = DecimalField(
        'Receiver Antenna Gain (Gr)',
        validators=[InputRequired(), non_negative_check]
    )
    receiver_antenna_gain_unit = SelectField(
        'Unit', choices=[('dB','dB'),('dBm','dBm'),('watt','Watt')],
        validators=[DataRequired()]
    )

    # Data rate (must be positive)
    data_rate = DecimalField(
        'Data Rate (kbps)',
        validators=[DataRequired(), positive_check]
    )

    # Feed-line and other link losses (can be zero or positive)
    antenna_feed_line_loss = DecimalField(
        'Antenna Feed Line Loss (Lf)',
        validators=[InputRequired(), non_negative_check]
    )
    antenna_feed_line_loss_unit = SelectField(
        'Unit', choices=[('dB','dB'),('dBm','dBm'),('watt','Watt')],
        validators=[DataRequired()]
    )

    other_losses = DecimalField(
        'Other Losses (Lo)',
        validators=[InputRequired(), non_negative_check]
    )
    other_losses_unit = SelectField(
        'Unit', choices=[('dB','dB'),('dBm','dBm'),('watt','Watt')],
        validators=[DataRequired()]
    )

    # Fade margin (can be zero or positive)
    fade_margin = DecimalField(
        'Fade Margin (Fm)',
        validators=[InputRequired(), non_negative_check]
    )
    fade_margin_unit = SelectField(
        'Unit', choices=[('dB','dB'),('dBm','dBm'),('watt','Watt')],
        validators=[DataRequired()]
    )

    # Amplifier gains (can be zero or positive)
    receiver_amplifier_gain = DecimalField(
        'Receiver Amplifier Gain (Ar)',
        validators=[InputRequired(), non_negative_check]
    )
    receiver_amplifier_gain_unit = SelectField(
        'Unit', choices=[('dB','dB'),('dBm','dBm'),('watt','Watt')],
        validators=[DataRequired()]
    )

    transmitter_amplifier_gain = FloatField(
        'Transmitter Amplifier Gain (At)',
        validators=[InputRequired(), non_negative_check]
    )
    transmitter_amplifier_gain_unit = SelectField(
        'Unit', choices=[('dB','dB'),('dBm','dBm'),('watt','Watt')],
        validators=[DataRequired()]
    )

    # Noise figure (positive)
    noise_figure_total = DecimalField(
        'Noise Figure Total',
        validators=[DataRequired(), positive_check]
    )
    noise_figure_total_unit = SelectField(
        'Unit', choices=[('dB','dB'),('dBm','dBm'),('watt','Watt')],
        validators=[DataRequired()]
    )

    # Noise temperature (>= 1 K)
    noise_temperature = DecimalField(
        'Noise Temperature (K)',
        validators=[DataRequired(), NumberRange(min=1, message='Must be at least 1 K')]
    )

    # Link margin (can be zero or positive)
    link_margin = DecimalField(
        'Link Margin',
        validators=[InputRequired(), non_negative_check]
    )
    link_margin_unit = SelectField(
        'Unit', choices=[('dB','dB'),('dBm','dBm'),('watt','Watt')],
        validators=[DataRequired()]
    )

    modulation = SelectField(
        'Modulation Technique',
        choices=[('BPSK','BPSK'),('QPSK','QPSK'),('8-PSK','8-PSK'),('16-PSK','16-PSK')],
        validators=[DataRequired()]
    )

    max_bit_error_rate = SelectField(
    'Maximum Bit Error Rate',
    choices=[('10^-1', '10^-1'), ('10^-1.5', '10^-1.5'), ('10^-2', '10^-2'), ('10^-2.5', '10^-2.5'), ('10^-3', '10^-3'), ('10^-3.5', '10^-3.5'), ('10^-4', '10^-4'), 
                ('10^-4.5', '10^-4.5'), ('10^-5','10^-5'), ('10^-5.5', '10^-5.5'), ('10^-6', '10^-6'), ('10^-6.5', '10^-6.5'), 
                ('10^-7','10^-7'), ('10^-7.5', '10^-7.5'),('10^-8','10^-8')],validators=[DataRequired()] )    
    

    submit = SubmitField('Calculate')

class CellularSystem(FlaskForm):
    # Network Configuration Parameters
    times_slots_per_carrier = IntegerField(
        'Times Slots per Carrier',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=1, max=1000, message="Must be between 1 and 1000")
        ],
        render_kw={
            "placeholder": "e.g., 8",
            "class": "form-control"
        }
    )
    
    area = FloatField(
        'Area (km²)',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=0.1, max=10000, message="Must be between 0.1 and 10,000 km²")
        ],
        render_kw={
            "placeholder": "e.g., 100.5",
            "class": "form-control",
            "step": "0.1"
        }
    )
    
    number_of_users = IntegerField(
        'Number of Users',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=1, max=1000000, message="Must be between 1 and 1,000,000")
        ],
        render_kw={
            "placeholder": "e.g., 5000",
            "class": "form-control"
        }
    )
    
    # Traffic Parameters
    average_calls_per_day = FloatField(
        'Average Calls per Day (λ)',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=0.1, max=1000, message="Must be between 0.1 and 1000")
        ],
        render_kw={
            "placeholder": "e.g., 2.5",
            "class": "form-control",
            "step": "0.1"
        }
    )
    
    average_call_duration = FloatField(
        'Average Call Duration (min)',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=0.1, max=300, message="Must be between 0.1 and 300 minutes")
        ],
        render_kw={
            "placeholder": "e.g., 3.5",
            "class": "form-control",
            "step": "0.1"
        }
    )
    
    call_drop_probability = FloatField(
        'Call Drop Probability',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=0, max=1, message="Must be between 0 and 1")
        ],
        render_kw={
            "placeholder": "e.g., 0.02",
            "class": "form-control",
            "step": "0.001"
        }
    )
    
    traffic_model = SelectField(
        'Traffic Model',
        choices=[
            ('erlang_b', 'Drop and Clear (Erlang B)'),
            ('erlang_c', 'Drop and Queue (Erlang C)')
        ],
        default='erlang_b',
        validators=[DataRequired(message="Please select a traffic model")],
        render_kw={
            "class": "form-control"
        }
    )
    
    # RF Parameters
    minimum_sir_required = FloatField(
        'Minimum SIR required (dB)',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=-50, max=50, message="Must be between -50 and 50 dB")
        ],
        render_kw={
            "placeholder": "e.g., 12.0",
            "class": "form-control",
            "step": "0.1"
        }
    )
    
    measured_power_at_reference_distance = FloatField(
        'Measured Power at Reference Distance (dB)',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=-200, max=100, message="Must be between -200 and 100 dB")
        ],
        render_kw={
            "placeholder": "e.g., -40.0",
            "class": "form-control",
            "step": "0.1"
        }
    )
    
    reference_distance = FloatField(
        'Reference Distance (meters)',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=1, max=10000, message="Must be between 1 and 10,000 meters")
        ],
        render_kw={
            "placeholder": "e.g., 100",
            "class": "form-control",
            "step": "1"
        }
    )
    
    path_loss_exponent = FloatField(
        'Path Loss Exponent',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=1.5, max=6, message="Must be between 1.5 and 6")
        ],
        render_kw={
            "placeholder": "e.g., 3.5",
            "class": "form-control",
            "step": "0.1"
        }
    )
    
    receiver_sensitivity = FloatField(
        'Receiver Sensitivity (μW)',
        validators=[
            DataRequired(message="This field is required"),
            NumberRange(min=0.001, max=1000, message="Must be between 0.001 and 1000 μW")
        ],
        render_kw={
            "placeholder": "e.g., 0.5",
            "class": "form-control",
            "step": "0.001"
        }
    )
    
    # Submit button
    submit = SubmitField(
        'Calculate',
        render_kw={
            "class": "btn btn-primary btn-lg btn-block"
        }
    )
    
    # Custom validation methods
    def validate_call_drop_probability(self, field):
        """Ensure call drop probability is a valid percentage"""
        if field.data is not None:
            if field.data < 0 or field.data > 1:
                raise ValidationError('Call drop probability must be between 0 and 1 (0% to 100%)')
    
    def validate_path_loss_exponent(self, field):
        """Ensure path loss exponent is realistic"""
        if field.data is not None:
            if field.data < 2 and field.data != 2:  # Free space is 2
                raise ValidationError('Path loss exponent should typically be 2 or higher')
    
    def validate_average_calls_per_day(self, field):
        """Ensure average calls per day is reasonable"""
        if field.data is not None:
            if field.data > 50:  # More than 50 calls per day seems unrealistic
                raise ValidationError('Average calls per day seems unusually high. Please verify.')
    
    def validate_receiver_sensitivity(self, field):
        """Ensure receiver sensitivity is in realistic range"""
        if field.data is not None:
            if field.data > 100:  # Very high sensitivity
                raise ValidationError('Receiver sensitivity seems unusually high. Please verify.')
