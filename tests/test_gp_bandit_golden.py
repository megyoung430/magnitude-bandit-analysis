"""Golden regression tests for the seeded GP bandit simulations.

The expected values were produced by the original (pre-refactor) implementation.
They pin the exact outputs *and* the generator state afterwards, so any change
in the order or number of random draws fails here.  Inputs are tiny and built
inline, so nothing depends on lab data.
"""

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tasks.gp_bandit import (
    plot_example_session_gp,
    sample_two_gp_bandits,
    simulate_3armed_blockswaps_gp,
    sweep_gp_metrics,
)

# Generator states are (PCG64 state, increment, has_uint32, uinteger).
GOLDEN = {
    "sample": {
        0: (
            [[3, 2], [3, 2], [3, 2], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [2, 3], [1, 2]],
            [
                [2.7514605679169764, 2.235790141312566],
                [3.0590013056592573, 2.2955612047289065],
                [3.2241108385828614, 2.424949472987117],
                [3.3193094903693314, 2.66257403784826],
                [3.3946575343099306, 2.9568430272221664],
                [3.442598816398005, 3.206162811587519],
                [3.3517842686399386, 3.3001764158334126],
                [2.9811732066127763, 3.170958351820506],
                [2.2466181569673553, 2.825751585096952],
                [1.2145994652366505, 2.3645506016514104],
            ],
            (179778097340537474880264210224901646943, 87136372517582989555478159403783844777, 0, 0),
        ),
        1: (
            [[3, 4], [3, 3], [4, 3], [4, 2], [4, 2], [4, 2], [4, 3], [4, 3], [3, 3], [3, 4]],
            [
                [3.1911687297136777, 4.143237108620255],
                [3.332575936367981, 3.4511391277739927],
                [3.568566543195352, 2.819604786568366],
                [3.80694083669, 2.3896095113761198],
                [3.9639954459066655, 2.2358539227117897],
                [3.993034064280165, 2.3586380500663675],
                [3.8771407423710427, 2.6777449478922493],
                [3.622340593734771, 3.08786306071657],
                [3.2543709541067987, 3.472823013454383],
                [2.7984824037459197, 3.742667230177942],
            ],
            (
                288514794402043327389427592626368753378,
                194290289479364712180083596243593368443,
                0,
                0,
            ),
        ),
        2: (
            [[3, 1], [3, 1], [3, 1], [3, 1], [4, 1], [4, 1], [4, 1], [4, 2], [4, 2], [4, 2]],
            [
                [2.8781069526404006, 1.4545025942901944],
                [2.663123863120652, 0.2847451707525921],
                [2.7552585229299247, -0.48875120656914506],
                [3.086757389024047, -0.6998785642533507],
                [3.522118114708454, -0.3700015289102221],
                [3.939629077233023, 0.31522813929969784],
                [4.247863833061103, 1.0758372215510246],
                [4.423123588173523, 1.6643001616948374],
                [4.48852483608176, 1.9338316689918282],
                [4.486545210378651, 1.8610598055714624],
            ],
            (
                288471395411178885866786362417739141184,
                121863417007658695389390353187995180015,
                0,
                0,
            ),
        ),
    },
    "sim": {
        (0, False): (
            [1.4166666666666667, 0.4166666666666667, 6.0, 2.1213203435596424],
            {
                "vals3": [
                    [3.0, 2.0, 0.0],
                    [3.0, 2.0, 0.0],
                    [3.0, 2.0, 0.0],
                    [3.0, 3.0, 0.0],
                    [0.0, 3.0, 3.0],
                    [0.0, 3.0, 3.0],
                    [0.0, 3.0, 3.0],
                    [0.0, 3.0, 3.0],
                    [3.0, 0.0, 3.0],
                    [3.0, 0.0, 2.0],
                    [3.0, 0.0, 2.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 4.0, 1.0],
                    [0.0, 4.0, 1.0],
                    [0.0, 4.0, 1.0],
                    [0.0, 4.0, 1.0],
                    [0.0, 4.0, 1.0],
                    [0.0, 4.0, 1.0],
                ],
                "boundaries": [4, 8, 15],
                "best_arm_series": [
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                "best_minus_second": [
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                ],
                "gp_int": [
                    [3, 2],
                    [3, 2],
                    [3, 2],
                    [3, 3],
                    [3, 3],
                    [3, 3],
                    [3, 3],
                    [3, 3],
                    [3, 3],
                    [2, 3],
                    [2, 3],
                    [1, 2],
                    [1, 2],
                    [1, 2],
                    [1, 3],
                    [1, 3],
                    [1, 3],
                    [1, 3],
                    [1, 4],
                    [1, 4],
                    [1, 4],
                    [1, 4],
                    [1, 4],
                    [1, 4],
                ],
                "gp_raw": [
                    [2.7514605679169764, 2.235790141312566],
                    [3.000115999530797, 2.282567066915859],
                    [3.1607080844353392, 2.3756929903943242],
                    [3.2683471859724564, 2.537999647096801],
                    [3.3444487148694746, 2.7459366104454896],
                    [3.395246968696374, 2.9660221874266357],
                    [3.3709739342431755, 3.1427268394152366],
                    [3.20872806074661, 3.218937413467695],
                    [2.8462405752192277, 3.1583697111295463],
                    [2.271177912900269, 2.9718109650148454],
                    [1.5261445553504684, 2.717300821482313],
                    [0.7056261428779398, 2.4791152288581646],
                    [-0.04924106409901485, 2.3467031026356726],
                    [-0.6274580231483675, 2.364487915914526],
                    [-0.952084332452416, 2.5369119772515254],
                    [-1.023424602885611, 2.806860220319295],
                    [-0.8923678390958831, 3.1014568595024485],
                    [-0.6427035635054783, 3.3483448503229196],
                    [-0.36734132599445246, 3.5016551550593222],
                    [-0.1138716616454829, 3.575745776984071],
                    [0.07815743239271011, 3.622799701907118],
                    [0.2383365404351938, 3.7159043206438853],
                    [0.39658344148641955, 3.9095722488565063],
                    [0.6161890833422119, 4.25258399781685],
                ],
            },
            (
                254078746502274486753339801108703095689,
                87136372517582989555478159403783844777,
                0,
                3420186697,
            ),
        ),
        (1, False): (
            [1.125, 0.3333333333333333, 4.0, 3.2145502536643185],
            {
                "vals3": [
                    [3.0, 4.0, 0.0],
                    [3.0, 4.0, 0.0],
                    [3.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [0.0, 3.0, 4.0],
                    [0.0, 3.0, 3.0],
                    [0.0, 4.0, 3.0],
                    [0.0, 4.0, 3.0],
                    [0.0, 4.0, 2.0],
                    [0.0, 4.0, 2.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 2.0, 1.0],
                    [0.0, 2.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                    [0.0, 2.0, 3.0],
                ],
                "boundaries": [8, 19, 23],
                "best_arm_series": [
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    2,
                ],
                "best_minus_second": [
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "gp_int": [
                    [3, 4],
                    [3, 4],
                    [3, 3],
                    [4, 3],
                    [4, 2],
                    [4, 2],
                    [4, 2],
                    [4, 3],
                    [4, 3],
                    [3, 3],
                    [3, 4],
                    [3, 4],
                    [2, 4],
                    [2, 4],
                    [1, 3],
                    [1, 3],
                    [1, 3],
                    [1, 2],
                    [1, 2],
                    [1, 2],
                    [2, 2],
                    [2, 2],
                    [2, 2],
                    [3, 2],
                ],
                "gp_raw": [
                    [3.1911687297136777, 4.143237108620255],
                    [3.308349297348409, 3.594590876413832],
                    [3.4890523201122163, 3.0740605853125156],
                    [3.68635849035214, 2.659182390768652],
                    [3.85682637902533, 2.404134835135676],
                    [3.9622827911481813, 2.336218904114374],
                    [3.9716050358194597, 2.4393341983268413],
                    [3.870104162659885, 2.6792682938043826],
                    [3.660767288761369, 2.991344065655377],
                    [3.3493145429001796, 3.307907310092321],
                    [2.963133740910049, 3.563277192145747],
                    [2.555603056102976, 3.726323069512504],
                    [2.138417697647505, 3.753403302313118],
                    [1.748580456728555, 3.6413442950749],
                    [1.3963724260111443, 3.398298636122556],
                    [1.119245554184558, 3.033078507461828],
                    [0.9344196240226239, 2.611534787842534],
                    [0.8871375402944293, 2.198494385617947],
                    [0.9931233716001497, 1.85164538721148],
                    [1.2561398308425031, 1.630066651552586],
                    [1.6228668315203365, 1.5379004998113404],
                    [2.022793624278663, 1.5394781673115434],
                    [2.365561585390446, 1.5780378879429442],
                    [2.5809846940511396, 1.5820562924886397],
                ],
            },
            (
                314839679854282588968503221014364209443,
                194290289479364712180083596243593368443,
                0,
                3764698162,
            ),
        ),
        (2, True): (
            [3.2904143810168893, 0.9166666666666666, 4.8, 2.039607805437114],
            {
                "vals3": [
                    [2.8781069526404006, 1.4545025942901944, 0.0],
                    [2.707028622846634, 0.508279699699645, 0.0],
                    [2.732445451501163, -0.19768089042785197, 0.0],
                    [2.9265935322029044, -0.5628752681503859, 0.0],
                    [3.231946045088506, -0.5669698753789634, 0.0],
                    [3.5898232167393114, -0.26172977575842626, 0.0],
                    [0.0, 0.23308094160949855, 3.9330366293878165],
                    [0.0, 0.7812698467942933, 4.22265718874349],
                    [0.0, 1.2499255189640228, 4.438941388107727],
                    [0.0, 1.5433659213740336, 4.579051035266274],
                    [4.664576240641826, 1.6300579955807364, 0.0],
                    [4.717481711047208, 1.5349902446331696, 0.0],
                    [4.7418019955992605, 1.3374406253901392, 0.0],
                    [4.769050060268054, 1.1077373791377234, 0.0],
                    [4.769644895123244, 0.9025102782265548, 0.0],
                    [4.748544421243095, 0.749186387047627, 0.0],
                    [4.700524220083863, 0.6472686840384028, 0.0],
                    [4.629406218445466, 0.5850522943050152, 0.0],
                    [0.0, 0.5717710789440384, 4.5471443708735615],
                    [0.0, 0.6180215002700833, 4.451030089910198],
                    [0.0, 0.7763810413316334, 4.3424079700481455],
                    [0.0, 1.0549294012739805, 4.192901619128887],
                    [3.97393784105743, 1.4719183919599452, 0.0],
                    [3.6613562103983597, 2.01105876683238, 0.0],
                ],
                "boundaries": [6, 10, 18, 22],
                "best_arm_series": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    2,
                    0,
                    0,
                ],
                "best_minus_second": [
                    1.4236043583502063,
                    2.198748923146989,
                    2.930126341929015,
                    3.4894688003532903,
                    3.7989159204674694,
                    3.8515529924977376,
                    3.699955687778318,
                    3.4413873419491967,
                    3.189015869143704,
                    3.0356851138922405,
                    3.0345182450610895,
                    3.1824914664140387,
                    3.4043613702091213,
                    3.6613126811303305,
                    3.8671346168966894,
                    3.9993580341954686,
                    4.0532555360454605,
                    4.0443539241404505,
                    3.975373291929523,
                    3.8330085896401145,
                    3.566026928716512,
                    3.137972217854906,
                    2.502019449097485,
                    1.6502974435659796,
                ],
                "gp_int": [
                    [3, 1],
                    [3, 1],
                    [3, 1],
                    [3, 1],
                    [3, 1],
                    [4, 1],
                    [4, 1],
                    [4, 1],
                    [4, 1],
                    [5, 2],
                    [5, 2],
                    [5, 2],
                    [5, 1],
                    [5, 1],
                    [5, 1],
                    [5, 1],
                    [5, 1],
                    [5, 1],
                    [5, 1],
                    [4, 1],
                    [4, 1],
                    [4, 1],
                    [4, 1],
                    [4, 2],
                ],
                "gp_raw": [
                    [2.8781069526404006, 1.4545025942901944],
                    [2.707028622846634, 0.508279699699645],
                    [2.732445451501163, -0.19768089042785197],
                    [2.9265935322029044, -0.5628752681503859],
                    [3.231946045088506, -0.5669698753789634],
                    [3.5898232167393114, -0.26172977575842626],
                    [3.9330366293878165, 0.23308094160949855],
                    [4.22265718874349, 0.7812698467942933],
                    [4.438941388107727, 1.2499255189640228],
                    [4.579051035266274, 1.5433659213740336],
                    [4.664576240641826, 1.6300579955807364],
                    [4.717481711047208, 1.5349902446331696],
                    [4.7418019955992605, 1.3374406253901392],
                    [4.769050060268054, 1.1077373791377234],
                    [4.769644895123244, 0.9025102782265548],
                    [4.748544421243095, 0.749186387047627],
                    [4.700524220083863, 0.6472686840384028],
                    [4.629406218445466, 0.5850522943050152],
                    [4.5471443708735615, 0.5717710789440384],
                    [4.451030089910198, 0.6180215002700833],
                    [4.3424079700481455, 0.7763810413316334],
                    [4.192901619128887, 1.0549294012739805],
                    [3.97393784105743, 1.4719183919599452],
                    [3.6613562103983597, 2.01105876683238],
                ],
            },
            (
                26155832263872595882597871344974113854,
                121863417007658695389390353187995180015,
                0,
                3289443787,
            ),
        ),
    },
    "sweep3": (
        [
            [[1.488888888888889, 1.1666666666666667], [2.055555555555556, 1.9444444444444444]],
            [[0.4444444444444445, 0.3], [0.5888888888888889, 0.6555555555555556]],
            [[5.277777777777778, 4.916666666666667], [6.166666666666667, 7.0]],
            [[2.754506041695658, 3.111621742825477], [4.416308344777451, 3.83047240179957]],
        ],
        "Done sig=4.0, max_block_len=12\n"
        "Done sig=9.0, max_block_len=12\n"
        "Done sig=4.0, max_block_len=15\n"
        "Done sig=9.0, max_block_len=15\n",
    ),
    "sweep11": (
        [
            [[1.6787878787878787, 1.5030303030303032], [1.8030303030303032, 1.5090909090909088]],
            [[0.4727272727272727, 0.4606060606060607], [0.5484848484848485, 0.4484848484848485]],
            [[4.453463203463203, 5.1309523809523805], [6.9805194805194795, 6.613636363636363]],
            [[2.539670414227993, 3.00507474376452], [2.8670961844038376, 3.29547418994273]],
        ],
        "Done sig=4.0, max_block_len=12\n"
        "Done sig=9.0, max_block_len=12\n"
        "Done sig=4.0, max_block_len=15\n"
        "Done sig=9.0, max_block_len=15\n",
    ),
    "sweep_raw": [
        [[2.0203916549006027, 1.7639585466503922], [2.786946297698657, 3.2495997132408014]],
        [[0.4444444444444445, 0.25555555555555554], [0.6444444444444444, 0.6555555555555556]],
        [[5.428571428571428, 5.833333333333333], [7.0, 7.0]],
        [[2.740532505855416, 3.3368818095710933], [4.584180290696357, 4.124551641152228]],
    ],
    "plot": {
        "title": "Example GP session | sig=5.00, max_block_len=20 | gap_mean=1.63, frac≥2=0.47, "
        "run_mean=10.00, run_std=2.94",
        "vals3": [
            [3.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [2.0, 3.0, 0.0],
            [2.0, 3.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 3.0, 0.0],
            [1.0, 3.0, 0.0],
            [1.0, 0.0, 3.0],
            [1.0, 0.0, 3.0],
            [1.0, 0.0, 4.0],
            [1.0, 0.0, 4.0],
            [1.0, 0.0, 4.0],
            [1.0, 0.0, 4.0],
            [1.0, 0.0, 4.0],
            [1.0, 0.0, 4.0],
            [1.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [2.0, 0.0, 5.0],
            [3.0, 0.0, 5.0],
            [4.0, 0.0, 5.0],
            [4.0, 0.0, 5.0],
        ],
        "boundaries": [16],
        "ylim": (-0.5, 5.5),
        "legend": ["Arm 0", "Arm 1", "Arm 2", "Block"],
    },
    "plot_raw": {
        "title": "Example GP session | sig=5.00, max_block_len=20 | gap_mean=2.16, "
        "frac≥2=0.53, run_mean=8.00, run_std=3.58",
        "ylim": (-0.5, 7.567614473119156),
        "ylabel": "Latent value",
    },
}


# Latent GP values pass through np.linalg.cholesky and a matrix product, whose last
# bits depend on which OpenBLAS kernel the CPU selects (about 1e-11 relative between
# kernels). Floats derived from them are compared to this relative tolerance; integer
# outputs, signs, key order and generator states stay exact.
BLAS_RTOL = 1e-9


def assert_blas_close(got, want) -> None:
    np.testing.assert_allclose(got, want, rtol=BLAS_RTOL, atol=0)


def generator_state(rng: np.random.Generator) -> tuple:
    state = rng.bit_generator.state
    return (state["state"]["state"], state["state"]["inc"], state["has_uint32"], state["uinteger"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sample_two_gp_bandits_matches_golden(seed):
    rng = np.random.default_rng(seed)
    gp_int, gp_raw = sample_two_gp_bandits(10, 4.0, rng=rng)
    expected_int, expected_raw, state = GOLDEN["sample"][seed]
    np.testing.assert_array_equal(gp_int, np.array(expected_int))
    assert_blas_close(gp_raw, np.array(expected_raw))
    assert gp_int.dtype == np.int64 and gp_raw.dtype == np.float64
    assert generator_state(rng) == state


@pytest.mark.parametrize(("seed", "use_raw"), [(0, False), (1, False), (2, True)])
def test_simulate_3armed_blockswaps_gp_matches_golden(seed, use_raw):
    rng = np.random.default_rng(seed)
    *metrics, session = simulate_3armed_blockswaps_gp(
        T=24,
        sig=5.0,
        max_block_len=12,
        min_block_len=4,
        n_blocks_draw=20,
        rng=rng,
        use_raw=use_raw,
        return_session=True,
    )
    expected_metrics, expected_session, state = GOLDEN["sim"][(seed, use_raw)]
    assert metrics == pytest.approx(expected_metrics, rel=BLAS_RTOL, abs=0)
    # Key order is part of the contract: callers may unpack the dict in order.
    assert list(session) == list(expected_session)
    for key, expected in expected_session.items():
        # Compare dtype and signs too: assert_allclose would accept -0.0 for the
        # inactive arm's +0.0 and a float32 copy of a float64 array.
        want = np.array(expected)
        got = session[key]
        assert got.dtype == want.dtype, key
        if np.issubdtype(want.dtype, np.floating):
            assert_blas_close(got, want)
            np.testing.assert_array_equal(np.signbit(got), np.signbit(want), err_msg=key)
        else:
            assert got.tobytes() == want.tobytes(), key
    assert generator_state(rng) == state


def assert_best_active_arm_becomes_inactive(session) -> None:
    """Each boundary sends the best *active* arm to zero and never a zero arm to best.

    Requires every active value to be non-zero, so the single zero in each row
    identifies the inactive arm.
    """
    vals3 = session["vals3"]
    T = len(vals3)
    zero_arms = (vals3 == 0).sum(axis=1)
    assert np.all(zero_arms == 1)
    inactive = np.argmax(vals3 == 0, axis=1)
    for stop in np.unique(session["boundaries"]):
        if stop >= T:
            continue
        last = vals3[stop - 1]
        active = [a for a in range(3) if a != inactive[stop - 1]]
        best_active = max(active, key=lambda a: last[a])
        assert inactive[stop] == best_active

    # Metrics compare the two active arms: the inactive arm's 0 never wins or
    # counts as second-best.
    rows = np.arange(T)
    best = session["best_arm_series"]
    assert np.all(best != inactive)
    active_sorted = np.sort(np.where(vals3 == 0, -np.inf, vals3), axis=1)
    np.testing.assert_array_equal(
        session["best_minus_second"], active_sorted[:, -1] - active_sorted[:, -2]
    )
    np.testing.assert_array_equal(vals3[rows, best], active_sorted[:, -1])


SWEEP_KWARGS = dict(
    sig_values=np.array([4.0, 9.0]), max_block_lens=np.array([12, 15]), T=30, seed=0
)


# With n_sims > 8 the cell mean differs in the last bit between plain left-to-right
# addition and np.mean's pairwise summation, so n_sims=11 pins the accumulation.
@pytest.mark.parametrize("n_sims", [3, 11])
def test_sweep_gp_metrics_matches_golden(n_sims, capsys):
    heatmaps = sweep_gp_metrics(n_sims=n_sims, **SWEEP_KWARGS)
    expected, prints = GOLDEN[f"sweep{n_sims}"]
    assert len(heatmaps) == 4
    for got, want in zip(heatmaps, expected, strict=True):
        np.testing.assert_array_equal(got, np.array(want))
    assert capsys.readouterr().out == prints


def test_sweep_gp_metrics_with_raw_values_matches_golden():
    heatmaps = sweep_gp_metrics(n_sims=3, use_raw=True, **SWEEP_KWARGS)
    for got, want in zip(heatmaps, GOLDEN["sweep_raw"], strict=True):
        assert_blas_close(got, np.array(want))


def test_sweep_returns_independent_arrays():
    heatmaps = sweep_gp_metrics(n_sims=1, **SWEEP_KWARGS)
    for first, second in itertools.combinations(heatmaps, 2):
        assert not np.shares_memory(first, second)


# n_sims is the divisor of every cell: a negative value divides an empty sum into
# -0.0 and 0 raises, so replacing it with the number of results would change both.
def test_sweep_gp_metrics_negative_n_sims_gives_negative_zero_heatmaps(capsys):
    heatmaps = sweep_gp_metrics(n_sims=-2, **SWEEP_KWARGS)
    for heatmap in heatmaps:
        assert heatmap.shape == (2, 2)
        assert np.all(heatmap == 0) and np.all(np.signbit(heatmap))
    assert capsys.readouterr().out == GOLDEN["sweep3"][1]


def test_sweep_gp_metrics_zero_n_sims_raises_before_printing(capsys):
    with pytest.raises(ZeroDivisionError):
        sweep_gp_metrics(n_sims=0, **SWEEP_KWARGS)
    assert capsys.readouterr().out == ""


# The inactive arm pays 0, which beats negative active values; roles must still be
# chosen among the active arms only.
def test_non_positive_values_assign_roles_among_active_arms():
    *_, session = simulate_3armed_blockswaps_gp(
        T=60,
        sig=5.0,
        max_block_len=8,
        min_block_len=2,
        shift=-3.0,
        use_raw=True,
        rng=np.random.default_rng(0),
        return_session=True,
    )
    assert len(np.unique(session["boundaries"])) > 3
    assert_best_active_arm_becomes_inactive(session)


def test_plot_example_session_gp_draws_expected_figure(monkeypatch):
    plt.close("all")
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    session = plot_example_session_gp(sig=5.0, max_block_len=20, T=30, seed=0)

    expected = GOLDEN["plot"]
    assert len(shown) == 1
    (ax,) = shown[0].axes
    n_boundaries = len(session["boundaries"])
    assert len(ax.lines) == 3 + n_boundaries
    assert ax.get_xlim() == (0, 29)
    assert ax.get_ylim() == expected["ylim"]
    assert ax.get_xlabel() == "Trial"
    assert ax.get_ylabel() == "Reward Magnitude"
    assert ax.get_title() == expected["title"]
    np.testing.assert_array_equal(session["vals3"], np.array(expected["vals3"]))
    np.testing.assert_array_equal(session["boundaries"], np.array(expected["boundaries"]))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == expected["legend"]
    plt.close("all")


def test_plot_example_session_gp_with_raw_values_labels_latent_axis(monkeypatch):
    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_example_session_gp(sig=5.0, max_block_len=20, T=40, seed=0, use_raw=True)
    ax = plt.gcf().axes[0]
    expected = GOLDEN["plot_raw"]
    assert ax.get_ylabel() == expected["ylabel"]
    assert ax.get_ylim() == pytest.approx(expected["ylim"], rel=BLAS_RTOL, abs=0)
    assert ax.get_title() == expected["title"]
    plt.close("all")


def test_plot_example_session_gp_bad_sig_fails_before_the_legend(monkeypatch):
    # The title is formatted from sig after the traces are drawn and before the legend.
    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda: None)
    with pytest.raises(ValueError, match="Unknown format code 'f'"):
        plot_example_session_gp(sig="5.0", max_block_len=20, T=30, seed=0)
    (ax,) = plt.gcf().axes
    assert len(ax.lines) >= 3
    assert ax.get_legend() is None
    plt.close("all")
