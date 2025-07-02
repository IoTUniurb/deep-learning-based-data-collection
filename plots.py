import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

PLOTS_DIR = "/home/l.calisti/notebooks/dlds_paper/plots"

# create plots dir
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot(
    xs: list,
    ys: list,
    save_path: str,
    labels: list[str] = None,
    xlabel: str = None,
    ylabel: str = None,
    fmts: list[str] = None,
    grid: bool = False,
    figsize: tuple[float] = (8, 4.5),
    xlim: tuple = None,
    ylim: tuple = None,
    colors: list = None,
    line_width: list[float] = None,
):
    assert len(xs) == len(
        ys
    ), f"xs and ys must have the same size. {len(xs)} != {len(ys)}"
    # assert len(xs) <= 5, f"not enough colors for {len(xs)} lines"

    default_colors = [
        (0.00, 0.45, 0.74),
        (0.85, 0.33, 0.10),
        (0.93, 0.69, 0.13),
        (0.49, 0.18, 0.56),
        (0.47, 0.67, 0.19),
    ]
    if colors is None:
        colors = default_colors

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(len(xs)):
        label = None
        if labels is not None:
            label = labels[i]
        fmt = "-o"
        if fmts is not None:
            fmt = fmts[i]
        lw = 2
        if line_width is not None:
            lw = line_width[i]

        ax.plot(xs[i], ys[i], fmt, label=label, color=colors[i], linewidth=lw)

    ax.tick_params(axis="both", which="major", labelsize=16)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontdict={"size": 18})
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontdict={"size": 18})

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if grid:
        ax.grid(color=(0.15, 0.15, 0.15), alpha=0.15)
        ax.minorticks_on()
        ax.grid(
            True,
            which="minor",
            linestyle="--",
            linewidth=0.5,
            color=(0.15, 0.15, 0.15),
            alpha=0.15,
        )
    if labels is not None:
        leg = ax.legend(fontsize=12)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_linestyle("-")
        leg.get_frame().set_boxstyle("Square")

    fig.tight_layout()
    plt.savefig(save_path)


def from_string(data: str, sep: str = "\t", header: int = None):
    return pd.read_csv(io.StringIO(data), sep=sep, header=header)


# ## DLBDC simulation

data = from_string(
    """
dataset	seed	model	window size	time steps	error	realign	alpha	tot samples	sensing count	inferences count	send count	skip count	error acc	error percent acc	Transmission Rate	Suppression Rate	Inference Rate	Sensing Rate	MAE	MAPE
co2_peano_no_weekend	69	model3	5	1	1	scaled-distance	0.5	5000	5000	4995	2324	2676	7600.379028	13.32849926	0.4648	0.5352	0.999	1	2.840201431	0.004980755
co2_peano_no_weekend	69	model3	5	1	3	scaled-distance	0.5	5000	5000	4995	884	4116	34195.04971	56.9547958	0.1768	0.8232	0.999	1	8.307835207	0.013837414
co2_peano_no_weekend	69	model3	5	1	5	scaled-distance	0.5	5000	5000	4995	545	4455	55387.40509	92.0481779	0.109	0.891	0.999	1	12.43263863	0.020661768
co2_peano_no_weekend	69	model3	5	1	7	scaled-distance	0.5	5000	5000	4995	425	4575	76003.00568	125.3711264	0.085	0.915	0.999	1	16.61267884	0.027403525
co2_peano_no_weekend	69	model3	5	1	10	scaled-distance	0.5	5000	5000	4995	297	4703	109022.1097	178.0843062	0.0594	0.9406	0.999	1	23.18139693	0.037866108
co2_peano_no_weekend	69	model3	5	1	25	scaled-distance	0.5	5000	5000	4995	105	4895	269170.8952	419.4926738	0.021	0.979	0.999	1	54.98894692	0.085698197
co2_peano_no_weekend	69	DBP	20	1	1	simple-append	1	5000	5000	4980	2322	2678	7290.714464	12.85091742	0.4644	0.5356	0.996	1	2.722447522	0.0047987
co2_peano_no_weekend	69	DBP	20	1	3	simple-append	1	5000	5000	4980	1066	3934	34338.66795	58.05610412	0.2132	0.7868	0.996	1	8.728690379	0.014757525
co2_peano_no_weekend	69	DBP	20	1	5	simple-append	1	5000	5000	4980	694	4306	67026.32274	111.2953144	0.1388	0.8612	0.996	1	15.5657972	0.025846566
co2_peano_no_weekend	69	DBP	20	1	7	simple-append	1	5000	5000	4980	487	4513	99905.05736	165.342842	0.0974	0.9026	0.996	1	22.13717203	0.036637014
co2_peano_no_weekend	69	DBP	20	1	10	simple-append	1	5000	5000	4980	320	4680	153051.7039	256.7759672	0.064	0.936	0.996	1	32.70335554	0.05486666
co2_peano_no_weekend	69	DBP	20	1	25	simple-append	1	5000	5000	4980	93	4907	356928.1248	594.1773332	0.0186	0.9814	0.996	1	72.73856223	0.121087698
co2_peano_no_weekend	69	KF	3	1	1	simple-append	1	5000	5000	4997	4427	573	1635.272549	3.124879365	0.8854	0.1146	0.9994	1	2.853878794	0.005453542
co2_peano_no_weekend	69	KF	3	1	3	simple-append	1	5000	5000	4997	3682	1318	12662.7508	23.98148553	0.7364	0.2636	0.9994	1	9.607549922	0.018195361
co2_peano_no_weekend	69	KF	3	1	5	simple-append	1	5000	5000	4997	3096	1904	33649.3037	62.40315532	0.6192	0.3808	0.9994	1	17.67295362	0.032774766
co2_peano_no_weekend	69	KF	3	1	7	simple-append	1	5000	5000	4997	2609	2391	62003.68997	113.0668147	0.5218	0.4782	0.9994	1	25.93211626	0.047288505
co2_peano_no_weekend	69	KF	3	1	10	simple-append	1	5000	5000	4997	2021	2979	115007.7575	206.4693899	0.4042	0.5958	0.9994	1	38.6061623	0.069308288
co2_peano_no_weekend	69	KF	3	1	25	simple-append	1	5000	5000	4997	439	4561	444141.4906	764.1709022	0.0878	0.9122	0.9994	1	97.37809485	0.167544596
noise_peano_no_weekend	69	model3	5	1	1	scaled-distance	1	5000	5000	4995	2530	2470	21.83578673	11.7604165	0.506	0.494	0.999	1	0.008840399	0.004761302
noise_peano_no_weekend	69	model3	5	1	3	scaled-distance	1	5000	5000	4995	1267	3733	90.59600858	47.72721845	0.2534	0.7466	0.999	1	0.024268955	0.012785218
noise_peano_no_weekend	69	model3	5	1	5	scaled-distance	1	5000	5000	4995	838	4162	164.9755616	88.64667716	0.1676	0.8324	0.999	1	0.03963853	0.021299057
noise_peano_no_weekend	69	model3	5	1	7	scaled-distance	1	5000	5000	4995	591	4409	247.2933461	132.7318041	0.1182	0.8818	0.999	1	0.056088307	0.030104741
noise_peano_no_weekend	69	model3	5	1	10	scaled-distance	1	5000	5000	4995	401	4599	361.1309841	193.7794709	0.0802	0.9198	0.999	1	0.078523806	0.042135132
noise_peano_no_weekend	69	model3	5	1	25	scaled-distance	1	5000	5000	4995	138	4862	875.2635167	475.4659131	0.0276	0.9724	0.999	1	0.180021291	0.097792249
noise_peano_no_weekend	69	DBP	20	1	1	simple-append	1	5000	5000	4980	2905	2095	17.82484353	9.595542681	0.581	0.419	0.996	1	0.008508279	0.004580211
noise_peano_no_weekend	69	DBP	20	1	3	simple-append	1	5000	5000	4980	1439	3561	87.7124779	46.78305438	0.2878	0.7122	0.996	1	0.024631418	0.013137617
noise_peano_no_weekend	69	DBP	20	1	5	simple-append	1	5000	5000	4980	992	4008	156.0242153	84.52640689	0.1984	0.8016	0.996	1	0.038928197	0.021089423
noise_peano_no_weekend	69	DBP	20	1	7	simple-append	1	5000	5000	4980	743	4257	224.67619	120.8250162	0.1486	0.8514	0.996	1	0.052778057	0.028382668
noise_peano_no_weekend	69	DBP	20	1	10	simple-append	1	5000	5000	4980	542	4458	327.955986	179.3621551	0.1084	0.8916	0.996	1	0.073565721	0.040233772
noise_peano_no_weekend	69	DBP	20	1	25	simple-append	1	5000	5000	4980	159	4841	767.9972802	417.2527373	0.0318	0.9682	0.996	1	0.158644346	0.086191435
noise_peano_no_weekend	69	KF	3	1	1	simple-append	1	5000	5000	4997	3694	1306	11.50727352	6.101048945	0.7388	0.2612	0.9994	1	0.008811082	0.004671554
noise_peano_no_weekend	69	KF	3	1	3	simple-append	1	5000	5000	4997	2388	2612	71.63103395	37.66187305	0.4776	0.5224	0.9994	1	0.027423826	0.014418788
noise_peano_no_weekend	69	KF	3	1	5	simple-append	1	5000	5000	4997	1847	3153	148.0954325	78.73274317	0.3694	0.6306	0.9994	1	0.04696969	0.02497074
noise_peano_no_weekend	69	KF	3	1	7	simple-append	1	5000	5000	4997	1491	3509	241.1992356	129.7930434	0.2982	0.7018	0.9994	1	0.068737314	0.036988613
noise_peano_no_weekend	69	KF	3	1	10	simple-append	1	5000	5000	4997	1094	3906	410.4725868	221.8265858	0.2188	0.7812	0.9994	1	0.105087708	0.056791241
noise_peano_no_weekend	69	KF	3	1	25	simple-append	1	5000	5000	4997	259	4741	1137.969094	645.8104251	0.0518	0.9482	0.9994	1	0.240027229	0.136218187
pm2p5_peano_no_weekend	69	model3	5	1	1	scaled-distance	1	5000	5000	4995	3900	1100	24.31994254	5.301444134	0.78	0.22	0.999	1	0.022109039	0.004819495
pm2p5_peano_no_weekend	69	model3	5	1	3	scaled-distance	1	5000	5000	4995	2366	2634	152.5690313	37.24725771	0.4732	0.5268	0.999	1	0.057922943	0.014140948
pm2p5_peano_no_weekend	69	model3	5	1	5	scaled-distance	1	5000	5000	4995	1538	3462	312.1118373	77.485087	0.3076	0.6924	0.999	1	0.090153621	0.022381596
pm2p5_peano_no_weekend	69	model3	5	1	7	scaled-distance	1	5000	5000	4995	1189	3811	458.7374149	117.8327512	0.2378	0.7622	0.999	1	0.120371927	0.030919116
pm2p5_peano_no_weekend	69	model3	5	1	10	scaled-distance	1	5000	5000	4995	838	4162	723.3275869	180.8477218	0.1676	0.8324	0.999	1	0.173793269	0.04345212
pm2p5_peano_no_weekend	69	model3	5	1	25	scaled-distance	1	5000	5000	4995	312	4688	2027.816716	501.0133741	0.0624	0.9376	0.999	1	0.43255476	0.106871454
pm2p5_peano_no_weekend	69	DBP	20	1	1	simple-append	1	5000	5000	4980	4265	735	16.37196484	3.714862313	0.853	0.147	0.996	1	0.022274782	0.005054234
pm2p5_peano_no_weekend	69	DBP	20	1	3	simple-append	1	5000	5000	4980	2935	2065	130.4340839	30.41267894	0.587	0.413	0.996	1	0.063164205	0.01472769
pm2p5_peano_no_weekend	69	DBP	20	1	5	simple-append	1	5000	5000	4980	2015	2985	285.8981862	70.20456606	0.403	0.597	0.996	1	0.095778287	0.023519118
pm2p5_peano_no_weekend	69	DBP	20	1	7	simple-append	1	5000	5000	4980	1461	3539	477.6555068	114.4903644	0.2922	0.7078	0.996	1	0.134969061	0.03235105
pm2p5_peano_no_weekend	69	DBP	20	1	10	simple-append	1	5000	5000	4980	991	4009	731.2006043	179.6182338	0.1982	0.8018	0.996	1	0.182389774	0.04480375
pm2p5_peano_no_weekend	69	DBP	20	1	25	simple-append	1	5000	5000	4980	310	4690	1978.54716	485.4987784	0.062	0.938	0.996	1	0.421865066	0.103517863
pm2p5_peano_no_weekend	69	KF	3	1	1	simple-append	1	5000	5000	4997	4699	301	5.564189527	1.494714983	0.9398	0.0602	0.9994	1	0.018485679	0.004965831
pm2p5_peano_no_weekend	69	KF	3	1	3	simple-append	1	5000	5000	4997	4170	830	47.34984401	12.27066039	0.834	0.166	0.9994	1	0.057048005	0.014783928
pm2p5_peano_no_weekend	69	KF	3	1	5	simple-append	1	5000	5000	4997	3695	1305	126.2839836	32.2383494	0.739	0.261	0.9994	1	0.096769336	0.024703716
pm2p5_peano_no_weekend	69	KF	3	1	7	simple-append	1	5000	5000	4997	3306	1694	233.231347	58.85679879	0.6612	0.3388	0.9994	1	0.137680842	0.034744273
pm2p5_peano_no_weekend	69	KF	3	1	10	simple-append	1	5000	5000	4997	2767	2233	465.9015397	115.3981135	0.5534	0.4466	0.9994	1	0.208643771	0.05167851
pm2p5_peano_no_weekend	69	KF	3	1	25	simple-append	1	5000	5000	4997	1303	3697	1913.442953	480.2376043	0.2606	0.7394	0.9994	1	0.517566392	0.129899271
rad_peano_no_weekend	69	model3	5	1	1	scaled-distance	1	5000	5000	4995	4407	593	706.207517	3.060006905	0.8814	0.1186	0.999	1	1.190906437	0.005160214
rad_peano_no_weekend	69	model3	5	1	3	scaled-distance	1	5000	5000	4995	3355	1645	6336.251872	23.99546967	0.671	0.329	0.999	1	3.851824846	0.014586912
rad_peano_no_weekend	69	model3	5	1	5	scaled-distance	1	5000	5000	4995	2506	2494	15814.87602	61.35305389	0.5012	0.4988	0.999	1	6.341169214	0.024600262
rad_peano_no_weekend	69	model3	5	1	7	scaled-distance	1	5000	5000	4995	1995	3005	25361.11783	99.72931142	0.399	0.601	0.999	1	8.439639875	0.033187791
rad_peano_no_weekend	69	model3	5	1	10	scaled-distance	1	5000	5000	4995	1523	3477	39577.41684	161.3231843	0.3046	0.6954	0.999	1	11.38263355	0.046397234
rad_peano_no_weekend	69	model3	5	1	25	scaled-distance	1	5000	5000	4995	738	4262	121609.9715	510.572323	0.1476	0.8524	0.999	1	28.53354565	0.119796416
rad_peano_no_weekend	69	DBP	20	1	1	simple-append	1	5000	5000	4980	4248	752	890.4560451	3.753265383	0.8496	0.1504	0.996	1	1.184117081	0.004991044
rad_peano_no_weekend	69	DBP	20	1	3	simple-append	1	5000	5000	4980	3006	1994	6366.154082	28.41272133	0.6012	0.3988	0.996	1	3.192655006	0.014249108
rad_peano_no_weekend	69	DBP	20	1	5	simple-append	1	5000	5000	4980	2055	2945	15079.91218	68.47254403	0.411	0.589	0.996	1	5.120513474	0.023250439
rad_peano_no_weekend	69	DBP	20	1	7	simple-append	1	5000	5000	4980	1456	3544	25295.80227	110.3619272	0.2912	0.7088	0.996	1	7.137641725	0.031140499
rad_peano_no_weekend	69	DBP	20	1	10	simple-append	1	5000	5000	4980	846	4154	42593.20613	176.5284969	0.1692	0.8308	0.996	1	10.25354023	0.042496027
rad_peano_no_weekend	69	DBP	20	1	25	simple-append	1	5000	5000	4980	302	4698	104703.8299	381.075865	0.0604	0.9396	0.996	1	22.2868944	0.081114488
rad_peano_no_weekend	69	KF	3	1	1	simple-append	1	5000	5000	4997	4474	526	449.6650166	2.572730874	0.8948	0.1052	0.9994	1	0.854876457	0.004891123
rad_peano_no_weekend	69	KF	3	1	3	simple-append	1	5000	5000	4997	3429	1571	3873.042207	23.51490707	0.6858	0.3142	0.9994	1	2.465335587	0.014968114
rad_peano_no_weekend	69	KF	3	1	5	simple-append	1	5000	5000	4997	2548	2452	9883.967766	59.07314836	0.5096	0.4904	0.9994	1	4.03098196	0.024091822
rad_peano_no_weekend	69	KF	3	1	7	simple-append	1	5000	5000	4997	1934	3066	17235.10516	98.9061475	0.3868	0.6132	0.9994	1	5.621365022	0.032259017
rad_peano_no_weekend	69	KF	3	1	10	simple-append	1	5000	5000	4997	1359	3641	29221.96448	158.9022093	0.2718	0.7282	0.9994	1	8.025807328	0.043642463
rad_peano_no_weekend	69	KF	3	1	25	simple-append	1	5000	5000	4997	615	4385	107624.6171	455.1719246	0.123	0.877	0.9994	1	24.54381234	0.103802035
    """,
    header=0,
)
data["error"] *= 0.01

# ### SR vs. error

# +
d = data[["dataset", "model", "error", "Suppression Rate"]]
d = d.drop(d[d["error"] == 0.25].index)
d = np.array(np.split(d.values, 4))

co2 = np.array(np.split(d[0], 3))
noise = np.array(np.split(d[1], 3))
pm = np.array(np.split(d[2], 3))
rad = np.array(np.split(d[3], 3))

plot(
    xs=co2[:, :, 2],
    ys=co2[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "co2_SR_error_dlbdc.pdf"),
    labels=["DLBDC", "DBP", "KF"],
    xlabel="$\epsilon$",
    ylabel="SR",
    grid=True,
)
plot(
    xs=noise[:, :, 2],
    ys=noise[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "noise_SR_error_dlbdc.pdf"),
    labels=["DLBDC", "DBP", "KF"],
    xlabel="$\epsilon$",
    ylabel="SR",
    grid=True,
)
plot(
    xs=pm[:, :, 2],
    ys=pm[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "pm_SR_error_dlbdc.pdf"),
    labels=["DLBDC", "DBP", "KF"],
    xlabel="$\epsilon$",
    ylabel="SR",
    grid=True,
)
plot(
    xs=rad[:, :, 2],
    ys=rad[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "rad_SR_error_dlbdc.pdf"),
    labels=["DLBDC", "DBP", "KF"],
    xlabel="$\epsilon$",
    ylabel="SR",
    grid=True,
)
# -

# ### MAPE vs. error

# +
d = data[["dataset", "model", "error", "MAPE"]]
d = d.drop(d[d["error"] == 0.25].index)
d = np.array(np.split(d.values, 4))

co2 = np.array(np.split(d[0], 3))
noise = np.array(np.split(d[1], 3))
pm = np.array(np.split(d[2], 3))
rad = np.array(np.split(d[3], 3))

plot(
    xs=co2[:, :, 2],
    ys=co2[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "co2_MAPE_error_dlbdc.pdf"),
    labels=["DLBDC", "DBP", "KF"],
    xlabel="$\epsilon$",
    ylabel="MAPE",
    grid=True,
    ylim=(0, 0.10),
)
plot(
    xs=noise[:, :, 2],
    ys=noise[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "noise_MAPE_error_dlbdc.pdf"),
    labels=["DLBDC", "DBP", "KF"],
    xlabel="$\epsilon$",
    ylabel="MAPE",
    grid=True,
    ylim=(0, 0.10),
)
plot(
    xs=pm[:, :, 2],
    ys=pm[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "pm_MAPE_error_dlbdc.pdf"),
    labels=["DLBDC", "DBP", "KF"],
    xlabel="$\epsilon$",
    ylabel="MAPE",
    grid=True,
    ylim=(0, 0.10),
)
plot(
    xs=rad[:, :, 2],
    ys=rad[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "rad_MAPE_error_dlbdc.pdf"),
    labels=["DLBDC", "DBP", "KF"],
    xlabel="$\epsilon$",
    ylabel="MAPE",
    grid=True,
    ylim=(0, 0.10),
)
# -

# ## DLDS validation

# +
data = from_string(
    """
dataset	model	seed	window size	time steps	MAE	MAPE	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15
co2_peano_no_weekend	model3	69	5	15	25.61951554	0.038912107	0.00910236	0.01318424	0.019069551	0.024936261	0.029670054	0.033986302	0.037782933	0.041459578	0.044761125	0.047612771	0.050468146	0.053493745	0.056355761	0.05936283	0.062435952
pm2p5_peano_no_weekend	model3	69	5	15	0.482061754	0.130774333	0.039644512	0.056181775	0.075165401	0.096303228	0.10689129	0.117717384	0.128528295	0.138408405	0.148238043	0.157327259	0.16482621	0.172278217	0.17950518	0.186540991	0.194058808
rad_peano_no_weekend	model3	69	5	15	78.64995096	0.240187248	0.058254795	0.098324696	0.135818638	0.18497174	0.222479738	0.246393033	0.261308345	0.272071053	0.280635373	0.287517037	0.294643385	0.302850368	0.310865958	0.31863549	0.328039075
noise_peano_no_weekend	model3	69	5	15	0.121344511	0.064070986	0.023546614	0.035703676	0.048131987	0.05808547	0.062475353	0.065726203	0.068025175	0.070135958	0.071727132	0.073114994	0.074628035	0.07575174	0.07685962	0.077980689	0.079172146
    """,
    header=0,
)
data = data[
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
].values
xs = (
    np.array(
        [
            np.arange(15),
            np.arange(15),
            np.arange(15),
            np.arange(15),
            np.arange(15),
            np.arange(15),
            np.arange(15),
            np.arange(15),
        ]
    )
    + 1
)


def linear(x: list, m: float, q: float):
    return x * m + q


ys = [
    data[0],
    linear(xs[0], 0.0037, 0.009),
    data[1],
    linear(xs[0], 0.0106, 0.0462),
    data[2],
    linear(xs[0], 0.0174, 0.1009),
    data[3],
    linear(xs[0], 0.0033, 0.038),
]

plot(
    ys=ys,
    xs=xs,
    save_path=os.path.join(PLOTS_DIR, "validate_dlds.pdf"),
    labels=["$CO_2$", "", "$PM_{2.5}$", "", "$\gamma~dose~rate$", "", "$noise$", ""],
    xlabel="steps",
    ylabel="MAPE",
    grid=True,
    fmts=["-o", "-", "-o", "-", "-o", "-", "-o", "-"],
    colors=[
        (0.00, 0.45, 0.74),
        (0.00, 0.18, 0.25),
        (0.85, 0.33, 0.10),
        (0.00, 0.00, 0.00),
        (0.93, 0.69, 0.13),
        (0.00, 0.00, 0.00),
        (0.49, 0.18, 0.56),
        (0.00, 0.00, 0.00),
    ],
    line_width=[2, 1, 2, 1, 2, 1, 2, 1],
)
# -

# ## DLDS simulation

# ## DLDS simulation
data = from_string(
    """
dataset	seed	model	window size	time steps	error	realign	alpha	tot samples	sensing count	inferences count	send count	skip count	error acc	error percent acc	Transmission Rate	Suppression Rate	Inference Rate	Sensing Rate	MAE	MAPE
co2_peano_no_weekend	69	model3	5	3	1	simple-append	1	5000	1669	1664	1410	3587	321939.6677	450.2435454	0.282	0.718	0.3328	0.3338	89.75178916	0.125520922
co2_peano_no_weekend	69	model3	5	5	1	simple-append	1	5000	1003	998	885	4110	290462.6705	419.0472817	0.177	0.823	0.1996	0.2006	70.6721826	0.101957976
co2_peano_no_weekend	69	model3	5	7	1	simple-append	1	5000	718	713	590	4406	212508.6786	319.1370882	0.118	0.882	0.1426	0.1436	48.23165652	0.072432385
co2_peano_no_weekend	69	model3	5	10	1	simple-append	1	5000	504	499	445	4550	315750.7703	444.114925	0.089	0.911	0.0998	0.1008	69.39577369	0.097607676
co2_peano_no_weekend	42	model3	5	15	1	simple-append	1	5000	337	332	313	4672	240275.4562	324.7594101	0.0626	0.9374	0.0664	0.0674	51.42882196	0.06951186
co2_peano_no_weekend	69	model3	5	3	3	simple-append	1	5000	1669	1664	1271	3726	336815.5114	476.5028544	0.2542	0.7458	0.3328	0.3338	90.39600413	0.127885898
co2_peano_no_weekend	69	model3	5	5	3	simple-append	1	5000	1003	998	679	4316	280393.857	405.5232311	0.1358	0.8642	0.1996	0.2006	64.96613925	0.093958117
co2_peano_no_weekend	69	model3	5	7	3	simple-append	1	5000	718	713	457	4539	217830.9604	324.9712805	0.0914	0.9086	0.1426	0.1436	47.99095845	0.071595347
co2_peano_no_weekend	69	model3	5	10	3	simple-append	1	5000	504	499	321	4674	298070.4634	425.2130677	0.0642	0.9358	0.0998	0.1008	63.77202897	0.090974127
co2_peano_no_weekend	42	model3	5	15	3	simple-append	1	5000	337	332	255	4730	245824.3225	334.5251077	0.051	0.949	0.0664	0.0674	51.97131554	0.070724124
co2_peano_no_weekend	69	model3	5	3	5	simple-append	1	5000	1669	1664	1155	3842	339270.9564	482.6343723	0.231	0.769	0.3328	0.3338	88.30581895	0.125620607
co2_peano_no_weekend	69	model3	5	5	5	simple-append	1	5000	1003	998	462	4533	263799.3669	372.7766121	0.0924	0.9076	0.1996	0.2006	58.19531589	0.082236182
co2_peano_no_weekend	69	model3	5	7	5	simple-append	1	5000	718	713	369	4627	211251.3746	320.2168432	0.0738	0.9262	0.1426	0.1436	45.65622965	0.069206147
co2_peano_no_weekend	69	model3	5	10	5	simple-append	1	5000	504	499	255	4740	306133.2199	437.2230199	0.051	0.949	0.0998	0.1008	64.58506749	0.092241143
co2_peano_no_weekend	42	model3	5	15	5	simple-append	1	5000	337	332	204	4781	248964.8854	342.04622	0.0408	0.9592	0.0664	0.0674	52.07380996	0.071542819
co2_peano_no_weekend	69	model3	5	3	7	simple-append	1	5000	1669	1664	1135	3862	357654.377	516.229945	0.227	0.773	0.3328	0.3338	92.60859062	0.133669069
co2_peano_no_weekend	69	model3	5	5	7	simple-append	1	5000	1003	998	370	4625	262745.3594	375.0695734	0.074	0.926	0.1996	0.2006	56.80980745	0.081096124
co2_peano_no_weekend	69	model3	5	7	7	simple-append	1	5000	718	713	288	4708	202722.947	312.4173885	0.0576	0.9424	0.1426	0.1436	43.05924957	0.066358834
co2_peano_no_weekend	69	model3	5	10	7	simple-append	1	5000	504	499	197	4798	298925.2035	436.8744234	0.0394	0.9606	0.0998	0.1008	62.30204325	0.091053444
co2_peano_no_weekend	42	model3	5	15	7	simple-append	1	5000	337	332	159	4826	266757.4487	372.4030749	0.0318	0.9682	0.0664	0.0674	55.27506189	0.077165991
co2_peano_no_weekend	69	model3	5	3	10	simple-append	1	5000	1669	1664	1003	3994	354325.0136	517.6826816	0.2006	0.7994	0.3328	0.3338	88.71432489	0.129615093
co2_peano_no_weekend	69	model3	5	5	10	simple-append	1	5000	1003	998	255	4740	255850.2499	380.2352825	0.051	0.949	0.1996	0.2006	53.97684597	0.080218414
co2_peano_no_weekend	69	model3	5	7	10	simple-append	1	5000	718	713	212	4784	211784.7899	338.143215	0.0424	0.9576	0.1426	0.1436	44.26939589	0.07068211
co2_peano_no_weekend	69	model3	5	10	10	simple-append	1	5000	504	499	141	4854	288765.5775	433.1200377	0.0282	0.9718	0.0998	0.1008	59.49023023	0.089229509
co2_peano_no_weekend	42	model3	5	15	10	simple-append	1	5000	337	332	113	4872	278604.4558	401.3114316	0.0226	0.9774	0.0664	0.0674	57.18482261	0.082370983
co2_peano_no_weekend	69	model3	5	3	25	simple-append	1	5000	1669	1664	61	4936	308391.5096	488.670475	0.0122	0.9878	0.3328	0.3338	62.47802059	0.099001312
co2_peano_no_weekend	69	model3	5	5	25	simple-append	1	5000	1003	998	91	4904	407955.8819	670.1012447	0.0182	0.9818	0.1996	0.2006	83.18839353	0.13664381
co2_peano_no_weekend	69	model3	5	7	25	simple-append	1	5000	718	713	122	4874	387189.5204	638.8623316	0.0244	0.9756	0.1426	0.1436	79.43978671	0.131075571
co2_peano_no_weekend	69	model3	5	10	25	simple-append	1	5000	504	499	44	4951	381221.5075	616.9294894	0.0088	0.9912	0.0998	0.1008	76.99889062	0.124607047
co2_peano_no_weekend	69	model3	5	15	25	simple-append	1	5000	337	332	48	4937	416847.2366	650.7216302	0.0096	0.9904	0.0664	0.0674	84.43330698	0.13180507
pm2p5_peano_no_weekend	69	model3	5	3	1	simple-append	1	5000	1669	1664	1572	3425	1586.594725	365.0223929	0.3144	0.6856	0.3328	0.3338	0.463239336	0.106575881
pm2p5_peano_no_weekend	69	model3	5	5	1	simple-append	1	5000	1003	998	964	4031	2924.754602	514.7551847	0.1928	0.8072	0.1996	0.2006	0.725565518	0.127699128
pm2p5_peano_no_weekend	69	model3	5	7	1	simple-append	1	5000	718	713	680	4316	2831.78857	635.8731157	0.136	0.864	0.1426	0.1436	0.656114126	0.147329267
pm2p5_peano_no_weekend	69	model3	5	10	1	simple-append	1	5000	504	499	488	4507	3506.347955	796.8603364	0.0976	0.9024	0.0998	0.1008	0.777978246	0.176805045
pm2p5_peano_no_weekend	69	model3	5	15	1	simple-append	1	5000	337	332	329	4656	4052.157855	1079.215448	0.0658	0.9342	0.0664	0.0674	0.870308818	0.231790259
pm2p5_peano_no_weekend	69	model3	5	3	3	simple-append	1	5000	1669	1664	1405	3592	1558.098634	362.6453885	0.281	0.719	0.3328	0.3338	0.433769108	0.100959184
pm2p5_peano_no_weekend	69	model3	5	5	3	simple-append	1	5000	1003	998	877	4118	2906.937515	517.2058256	0.1754	0.8246	0.1996	0.2006	0.705910033	0.125596364
pm2p5_peano_no_weekend	69	model3	5	7	3	simple-append	1	5000	718	713	612	4384	3046.971167	673.5477938	0.1224	0.8776	0.1426	0.1436	0.695020795	0.153637727
pm2p5_peano_no_weekend	69	model3	5	10	3	simple-append	1	5000	504	499	453	4542	3608.872035	810.7181493	0.0906	0.9094	0.0998	0.1008	0.79455571	0.178493648
pm2p5_peano_no_weekend	69	model3	5	15	3	simple-append	1	5000	337	332	311	4674	4137.466883	1095.233611	0.0622	0.9378	0.0664	0.0674	0.885209004	0.234324692
pm2p5_peano_no_weekend	69	model3	5	3	5	simple-append	1	5000	1669	1664	1226	3771	1596.120303	376.5653661	0.2452	0.7548	0.3328	0.3338	0.423261815	0.099858225
pm2p5_peano_no_weekend	69	model3	5	5	5	simple-append	1	5000	1003	998	777	4218	2988.340988	524.8345105	0.1554	0.8446	0.1996	0.2006	0.708473444	0.124427338
pm2p5_peano_no_weekend	69	model3	5	7	5	simple-append	1	5000	718	713	545	4451	2960.897271	664.0641718	0.109	0.891	0.1426	0.1436	0.665220685	0.149194377
pm2p5_peano_no_weekend	69	model3	5	10	5	simple-append	1	5000	504	499	414	4581	3524.558097	806.0409118	0.0828	0.9172	0.0998	0.1008	0.769386181	0.175953048
pm2p5_peano_no_weekend	69	model3	5	15	5	simple-append	1	5000	337	332	298	4687	4228.63304	1118.418657	0.0596	0.9404	0.0664	0.0674	0.902204617	0.238621433
pm2p5_peano_no_weekend	69	model3	5	3	7	simple-append	1	5000	1669	1664	1039	3958	1649.921466	384.808737	0.2078	0.7922	0.3328	0.3338	0.416857369	0.097223026
pm2p5_peano_no_weekend	69	model3	5	5	7	simple-append	1	5000	1003	998	665	4330	2785.016542	510.6026791	0.133	0.867	0.1996	0.2006	0.643190887	0.117922097
pm2p5_peano_no_weekend	69	model3	5	7	7	simple-append	1	5000	718	713	494	4502	2971.676751	670.2143329	0.0988	0.9012	0.1426	0.1436	0.660079243	0.148870354
pm2p5_peano_no_weekend	69	model3	5	10	7	simple-append	1	5000	504	499	374	4621	3595.762229	826.0695664	0.0748	0.9252	0.0998	0.1008	0.778135085	0.178764243
pm2p5_peano_no_weekend	69	model3	5	15	7	simple-append	1	5000	337	332	284	4701	4188.23724	1114.563796	0.0568	0.9432	0.0664	0.0674	0.890924748	0.237090788
pm2p5_peano_no_weekend	69	model3	5	3	10	simple-append	1	5000	1669	1664	824	4173	1841.882799	402.3496693	0.1648	0.8352	0.3328	0.3338	0.441380973	0.096417366
pm2p5_peano_no_weekend	69	model3	5	5	10	simple-append	1	5000	1003	998	542	4453	3009.105955	540.9563518	0.1084	0.8916	0.1996	0.2006	0.675748025	0.121481328
pm2p5_peano_no_weekend	69	model3	5	7	10	simple-append	1	5000	718	713	420	4576	2844.859625	674.3316616	0.084	0.916	0.1426	0.1436	0.621691352	0.147362688
pm2p5_peano_no_weekend	69	model3	5	10	10	simple-append	1	5000	504	499	339	4656	3697.196003	854.5371968	0.0678	0.9322	0.0998	0.1008	0.794071306	0.183534621
pm2p5_peano_no_weekend	69	model3	5	15	10	simple-append	1	5000	337	332	255	4730	4062.664543	1084.85995	0.051	0.949	0.0664	0.0674	0.85891428	0.229357283
pm2p5_peano_no_weekend	69	model3	5	3	25	simple-append	1	5000	1669	1664	294	4703	2392.551238	581.2735224	0.0588	0.9412	0.3328	0.3338	0.508728734	0.123596326
pm2p5_peano_no_weekend	69	model3	5	5	25	simple-append	1	5000	1003	998	197	4798	2966.804929	724.032083	0.0394	0.9606	0.1996	0.2006	0.618342003	0.150902893
pm2p5_peano_no_weekend	69	model3	5	7	25	simple-append	1	5000	718	713	245	4751	3826.512271	871.8672105	0.049	0.951	0.1426	0.1436	0.80541197	0.183512358
pm2p5_peano_no_weekend	69	model3	5	10	25	simple-append	1	5000	504	499	173	4822	4156.396354	961.9905763	0.0346	0.9654	0.0998	0.1008	0.861965233	0.199500327
pm2p5_peano_no_weekend	69	model3	5	15	25	simple-append	1	5000	337	332	148	4837	4310.919284	1150.383034	0.0296	0.9704	0.0664	0.0674	0.891238223	0.23782986
rad_peano_no_weekend	69	model3	5	3	1	simple-append	1	5000	1669	1664	1518	3479	145248.5615	309.352359	0.3036	0.6964	0.3328	0.3338	41.75008955	0.088919908
rad_peano_no_weekend	69	model3	5	5	1	simple-append	1	5000	1003	998	973	4022	289933.3555	866.7011249	0.1946	0.8054	0.1996	0.2006	72.08686113	0.215490086
rad_peano_no_weekend	69	model3	5	7	1	simple-append	1	5000	718	713	690	4306	279588.3918	839.4667261	0.138	0.862	0.1426	0.1436	64.92995629	0.194952793
rad_peano_no_weekend	69	model3	5	10	1	simple-append	1	5000	504	499	481	4514	485583.9005	1750.777799	0.0962	0.9038	0.0998	0.1008	107.5728623	0.387855073
rad_peano_no_weekend	69	model3	5	15	1	simple-append	1	5000	337	332	328	4657	361529.0558	1335.138579	0.0656	0.9344	0.0664	0.0674	77.6313197	0.286694992
rad_peano_no_weekend	69	model3	5	3	3	simple-append	1	5000	1669	1664	1198	3799	144297.9297	311.1783359	0.2396	0.7604	0.3328	0.3338	37.98313496	0.081910591
rad_peano_no_weekend	69	model3	5	5	3	simple-append	1	5000	1003	998	923	4072	289924.2471	866.1015796	0.1846	0.8154	0.1996	0.2006	71.19947129	0.212696852
rad_peano_no_weekend	69	model3	5	7	3	simple-append	1	5000	718	713	623	4373	279604.9218	839.8726399	0.1246	0.8754	0.1426	0.1436	63.93892562	0.192058687
rad_peano_no_weekend	69	model3	5	10	3	simple-append	1	5000	504	499	405	4590	485189.6657	1757.53485	0.081	0.919	0.0998	0.1008	105.7058095	0.382905196
rad_peano_no_weekend	69	model3	5	15	3	simple-append	1	5000	337	332	322	4663	361514.4971	1335.088419	0.0644	0.9356	0.0664	0.0674	77.52830734	0.286315337
rad_peano_no_weekend	69	model3	5	3	5	simple-append	1	5000	1669	1664	942	4055	149547.7198	338.2596673	0.1884	0.8116	0.3328	0.3338	36.87983225	0.08341792
rad_peano_no_weekend	69	model3	5	5	5	simple-append	1	5000	1003	998	865	4130	289681.6877	869.7569362	0.173	0.827	0.1996	0.2006	70.14084449	0.2105949
rad_peano_no_weekend	69	model3	5	7	5	simple-append	1	5000	718	713	539	4457	280930.7423	847.8449828	0.1078	0.8922	0.1426	0.1436	63.03135345	0.190227728
rad_peano_no_weekend	69	model3	5	10	5	simple-append	1	5000	504	499	359	4636	485702.4909	1761.119536	0.0718	0.9282	0.0998	0.1008	104.7675778	0.379879106
rad_peano_no_weekend	69	model3	5	15	5	simple-append	1	5000	337	332	304	4681	362413.6137	1340.596057	0.0608	0.9392	0.0664	0.0674	77.42226313	0.286390954
rad_peano_no_weekend	69	model3	5	3	7	simple-append	1	5000	1669	1664	726	4271	151427.7179	362.5616983	0.1452	0.8548	0.3328	0.3338	35.45486254	0.084889182
rad_peano_no_weekend	69	model3	5	5	7	simple-append	1	5000	1003	998	809	4186	282492.8905	871.5848785	0.1618	0.8382	0.1996	0.2006	67.48516256	0.208214257
rad_peano_no_weekend	69	model3	5	7	7	simple-append	1	5000	718	713	476	4520	277899.8885	850.534599	0.0952	0.9048	0.1426	0.1436	61.48227621	0.188171371
rad_peano_no_weekend	69	model3	5	10	7	simple-append	1	5000	504	499	302	4693	495175.6422	1766.556017	0.0604	0.9396	0.0998	0.1008	105.5136676	0.376423613
rad_peano_no_weekend	69	model3	5	15	7	simple-append	1	5000	337	332	294	4691	363904.5987	1349.586481	0.0588	0.9412	0.0664	0.0674	77.57505836	0.287696969
rad_peano_no_weekend	69	model3	5	3	10	simple-append	1	5000	1669	1664	516	4481	164525.6194	398.6714938	0.1032	0.8968	0.3328	0.3338	36.71627301	0.088969314
rad_peano_no_weekend	69	model3	5	5	10	simple-append	1	5000	1003	998	731	4264	290939.2872	895.9338302	0.1462	0.8538	0.1996	0.2006	68.23154015	0.210115814
rad_peano_no_weekend	69	model3	5	7	10	simple-append	1	5000	718	713	380	4616	283406.1545	875.3474785	0.076	0.924	0.1426	0.1436	61.39648062	0.189633336
rad_peano_no_weekend	69	model3	5	10	10	simple-append	1	5000	504	499	239	4756	494681.8689	1755.519185	0.0478	0.9522	0.0998	0.1008	104.0121676	0.369116734
rad_peano_no_weekend	69	model3	5	15	10	simple-append	1	5000	337	332	278	4707	368465.4131	1376.195482	0.0556	0.9444	0.0664	0.0674	78.28030872	0.292372102
rad_peano_no_weekend	69	model3	5	3	25	simple-append	1	5000	1669	1664	140	4857	179919.8748	533.685611	0.028	0.972	0.3328	0.3338	37.04341667	0.109879681
rad_peano_no_weekend	69	model3	5	5	25	simple-append	1	5000	1003	998	431	4564	317693.568	1158.143068	0.0862	0.9138	0.1996	0.2006	69.60858195	0.25375615
rad_peano_no_weekend	69	model3	5	7	25	simple-append	1	5000	718	713	127	4869	311937.5378	1067.892489	0.0254	0.9746	0.1426	0.1436	64.06603774	0.219324808
rad_peano_no_weekend	69	model3	5	10	25	simple-append	1	5000	504	499	106	4889	506052.6695	1904.567853	0.0212	0.9788	0.0998	0.1008	103.5084208	0.389561843
rad_peano_no_weekend	69	model3	5	15	25	simple-append	1	5000	337	332	185	4800	409446.9423	1593.104702	0.037	0.963	0.0664	0.0674	85.30144632	0.331896813
noise_peano_no_weekend	69	model3	5	3	1	simple-append	1	5000	1669	1664	1364	3633	298.8074896	147.559549	0.2728	0.7272	0.3328	0.3338	0.082248139	0.040616446
noise_peano_no_weekend	69	model3	5	5	1	simple-append	1	5000	1003	998	844	4151	399.3513003	198.6358049	0.1688	0.8312	0.1996	0.2006	0.096206047	0.047852519
noise_peano_no_weekend	69	model3	5	7	1	simple-append	1	5000	718	713	583	4413	506.4262499	248.6879124	0.1166	0.8834	0.1426	0.1436	0.114757818	0.056353481
noise_peano_no_weekend	69	model3	5	10	1	simple-append	1	5000	504	499	447	4548	683.947597	345.550944	0.0894	0.9106	0.0998	0.1008	0.150384256	0.07597866
noise_peano_no_weekend	69	model3	5	15	1	simple-append	1	5000	337	332	328	4657	1111.913298	528.4472407	0.0656	0.9344	0.0664	0.0674	0.238761713	0.113473747
noise_peano_no_weekend	69	model3	5	3	3	simple-append	1	5000	1669	1664	916	4081	324.4662212	163.3957399	0.1832	0.8168	0.3328	0.3338	0.079506548	0.040038162
noise_peano_no_weekend	69	model3	5	5	3	simple-append	1	5000	1003	998	607	4388	424.3089062	213.8598862	0.1214	0.8786	0.1996	0.2006	0.096697563	0.04873744
noise_peano_no_weekend	69	model3	5	7	3	simple-append	1	5000	718	713	426	4570	534.7410124	265.3590342	0.0852	0.9148	0.1426	0.1436	0.117011162	0.058065434
noise_peano_no_weekend	69	model3	5	10	3	simple-append	1	5000	504	499	372	4623	691.187922	350.7866125	0.0744	0.9256	0.0998	0.1008	0.14951069	0.075878566
noise_peano_no_weekend	69	model3	5	15	3	simple-append	1	5000	337	332	302	4683	1117.450107	531.9621923	0.0604	0.9396	0.0664	0.0674	0.23861843	0.113594318
noise_peano_no_weekend	69	model3	5	3	5	simple-append	1	5000	1669	1664	702	4295	361.5058619	183.8327151	0.1404	0.8596	0.3328	0.3338	0.084169002	0.042801563
noise_peano_no_weekend	69	model3	5	5	5	simple-append	1	5000	1003	998	471	4524	464.2260308	237.8184113	0.0942	0.9058	0.1996	0.2006	0.102614065	0.052568172
noise_peano_no_weekend	69	model3	5	7	5	simple-append	1	5000	718	713	339	4657	579.976247	290.2099896	0.0678	0.9322	0.1426	0.1436	0.124538597	0.06231694
noise_peano_no_weekend	69	model3	5	10	5	simple-append	1	5000	504	499	323	4672	720.4895193	363.9932969	0.0646	0.9354	0.0998	0.1008	0.154214366	0.077909524
noise_peano_no_weekend	69	model3	5	15	5	simple-append	1	5000	337	332	277	4708	1128.841046	539.8896834	0.0554	0.9446	0.0664	0.0674	0.239770825	0.114674954
noise_peano_no_weekend	69	model3	5	3	7	simple-append	1	5000	1669	1664	511	4486	401.2557929	208.6422407	0.1022	0.8978	0.3328	0.3338	0.089446231	0.046509639
noise_peano_no_weekend	69	model3	5	5	7	simple-append	1	5000	1003	998	372	4623	507.2657394	264.3290728	0.0744	0.9256	0.1996	0.2006	0.109726528	0.057176957
noise_peano_no_weekend	69	model3	5	7	7	simple-append	1	5000	718	713	283	4713	658.9071716	332.0875015	0.0566	0.9434	0.1426	0.1436	0.139806317	0.07046202
noise_peano_no_weekend	69	model3	5	10	7	simple-append	1	5000	504	499	286	4709	796.6604264	402.4834659	0.0572	0.9428	0.0998	0.1008	0.16917826	0.085471112
noise_peano_no_weekend	69	model3	5	15	7	simple-append	1	5000	337	332	244	4741	1164.75786	561.6206042	0.0488	0.9512	0.0664	0.0674	0.245677676	0.118460368
noise_peano_no_weekend	69	model3	5	3	10	simple-append	1	5000	1669	1664	349	4648	535.6350702	277.1650619	0.0698	0.9302	0.3328	0.3338	0.115239903	0.059631037
noise_peano_no_weekend	69	model3	5	5	10	simple-append	1	5000	1003	998	302	4693	608.9966646	320.9410651	0.0604	0.9396	0.1996	0.2006	0.129767028	0.068387186
noise_peano_no_weekend	69	model3	5	7	10	simple-append	1	5000	718	713	230	4766	782.1703698	398.7513907	0.046	0.954	0.1426	0.1436	0.164114639	0.083665839
noise_peano_no_weekend	69	model3	5	10	10	simple-append	1	5000	504	499	233	4762	889.1311868	454.5645967	0.0466	0.9534	0.0998	0.1008	0.186713815	0.095456656
noise_peano_no_weekend	69	model3	5	15	10	simple-append	1	5000	337	332	203	4782	1184.472925	583.2144952	0.0406	0.9594	0.0664	0.0674	0.247694045	0.121960371
noise_peano_no_weekend	69	model3	5	3	25	simple-append	1	5000	1669	1664	92	4905	1203.979913	635.4460585	0.0184	0.9816	0.3328	0.3338	0.245459717	0.129550675
noise_peano_no_weekend	69	model3	5	5	25	simple-append	1	5000	1003	998	118	4877	1067.304836	579.6511812	0.0236	0.9764	0.1996	0.2006	0.218844543	0.118854046
noise_peano_no_weekend	69	model3	5	7	25	simple-append	1	5000	718	713	80	4916	1407.837733	735.2822426	0.016	0.984	0.1426	0.1436	0.286378709	0.149569211
noise_peano_no_weekend	69	model3	5	10	25	simple-append	1	5000	504	499	69	4926	1543.110144	799.816323	0.0138	0.9862	0.0998	0.1008	0.313258251	0.162366286
noise_peano_no_weekend	69	model3	5	15	25	simple-append	1	5000	337	332	46	4939	1181.676118	668.0906155	0.0092	0.9908	0.0664	0.0674	0.239254124	0.135268398
    """,
    header=0,
)
data["error"] *= 0.01

# ### SR vs. error

# +
d = data[["dataset", "time steps", "error", "Suppression Rate"]]
d = d.drop(d[d["error"] == 0.25].index)
d = np.array(np.split(d.values, 4))
co2 = np.array(np.split(d[0], 5))
pm = np.array(np.split(d[1], 5))
rad = np.array(np.split(d[2], 5))
noise = np.array(np.split(d[3], 5))

plot(
    xs=co2[:, :, 1],
    ys=co2[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "co2_SR_error_dlds.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="SR",
    grid=True,
)
plot(
    xs=noise[:, :, 1],
    ys=noise[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "noise_SR_error_dlds.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="SR",
    grid=True,
)
plot(
    xs=pm[:, :, 1],
    ys=pm[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "pm_SR_error_dlds.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="SR",
    grid=True,
)
plot(
    xs=rad[:, :, 1],
    ys=rad[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "rad_SR_error_dlds.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="SR",
    grid=True,
)
# -

# ### MAPE vs. error

# +
d = data[["dataset", "time steps", "error", "MAPE"]]
d = d.drop(d[d["error"] == 0.25].index)
d = np.array(np.split(d.values, 4))
co2 = np.array(np.split(d[0], 5))
pm = np.array(np.split(d[1], 5))
rad = np.array(np.split(d[2], 5))
noise = np.array(np.split(d[3], 5))

plot(
    xs=co2[:, :, 1],
    ys=co2[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "co2_MAPE_error_dlds.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="MAPE",
    grid=True,
)
plot(
    xs=noise[:, :, 1],
    ys=noise[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "noise_MAPE_error_dlds.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="MAPE",
    grid=True,
)
plot(
    xs=pm[:, :, 1],
    ys=pm[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "pm_MAPE_error_dlds.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="MAPE",
    grid=True,
)
plot(
    xs=rad[:, :, 1],
    ys=rad[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "rad_MAPE_error_dlds.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="MAPE",
    grid=True,
)
# -
# ## DLDS simulation - lerp


data = from_string(
    """
dataset	seed	model	window size	time steps	error	realign	alpha	tot samples	sensing count	inferences count	send count	skip count	error acc	error percent acc	Transmission Rate	Suppression Rate	Inference Rate	Sensing Rate	MAE	MAPE	Active Energy [J]	Slepping energy [J]	Total Energy [J]	Active Energy Ration	Sleeping Energy Ratio	Total Energy Ratio
co2_peano_no_weekend	69	model3	5	3	1	lerp	1	5000	1669	1664	934	4063	26515.29773	41.51336584	0.1868	0.8132	0.3328	0.3338	6.526039313	0.010217417	1093.647816	43763.75943	44857.40725	3.540721893	0.991884531	1.054026563
co2_peano_no_weekend	69	model3	5	5	1	lerp	1	5000	1003	998	705	4290	51316.06241	79.28684128	0.141	0.859	0.1996	0.2006	11.96178611	0.018481781	703.7052312	43824.7398	44528.44503	5.502734092	0.990504364	1.061813381
co2_peano_no_weekend	69	model3	5	7	1	lerp	1	5000	718	713	563	4433	73314.58499	111.8129761	0.1126	0.8874	0.1426	0.1436	16.53836792	0.025222869	522.5943662	43851.72437	44374.31874	7.409767531	0.989894848	1.065501401
co2_peano_no_weekend	69	model3	5	10	1	lerp	1	5000	504	499	431	4564	105267.2128	157.9488911	0.0862	0.9138	0.0998	0.1008	23.0646829	0.034607557	378.3887313	43872.49933	44250.88806	10.23366302	0.989426102	1.068473444
co2_peano_no_weekend	69	model3	5	15	1	lerp	1	5000	337	332	283	4702	131726.4623	198.7403682	0.0566	0.9434	0.0664	0.0674	28.01498561	0.042267199	251.2916842	43889.62086	44140.91255	15.40959375	0.989040123	1.071135508
co2_peano_no_weekend	69	model3	5	3	3	lerp	1	5000	1669	1664	454	4543	44847.61124	72.58778318	0.0908	0.9092	0.3328	0.3338	9.871805247	0.01597794	938.2796206	43773.46063	44711.74025	4.12702427	0.991664707	1.057460491
co2_peano_no_weekend	69	model3	5	5	3	lerp	1	5000	1003	998	431	4564	65625.86398	104.3976394	0.0862	0.9138	0.1996	0.2006	14.37902366	0.022874154	615.0158862	43830.27757	44445.29345	6.296264622	0.990379218	1.063799901
co2_peano_no_weekend	69	model3	5	7	3	lerp	1	5000	718	713	362	4634	84876.29361	132.9311318	0.0724	0.9276	0.1426	0.1436	18.31598913	0.028686045	457.5339342	43855.78675	44313.32068	8.463422004	0.989803153	1.066968082
co2_peano_no_weekend	69	model3	5	10	3	lerp	1	5000	504	499	298	4697	116669.9934	179.0917325	0.0596	0.9404	0.0998	0.1008	24.8392577	0.038128962	335.3387938	43875.18737	44210.52616	11.54743453	0.989365485	1.069448905
co2_peano_no_weekend	69	model3	5	15	3	lerp	1	5000	337	332	202	4783	134449.6521	206.5713482	0.0404	0.9596	0.0664	0.0674	28.10990008	0.043188657	225.0733012	43891.25794	44116.33124	17.20462954	0.989003233	1.071732337
co2_peano_no_weekend	69	model3	5	3	5	lerp	1	5000	1669	1664	290	4707	66741.32437	109.4131711	0.058	0.942	0.3328	0.3338	14.17916388	0.023244778	885.1954871	43776.7752	44661.97069	4.374517068	0.991589623	1.058638883
co2_peano_no_weekend	69	model3	5	5	5	lerp	1	5000	1003	998	307	4688	85768.83411	138.4088853	0.0614	0.9386	0.1996	0.2006	18.29539977	0.02952408	574.8791023	43832.78371	44407.66281	6.735855853	0.990322593	1.064701355
co2_peano_no_weekend	69	model3	5	7	5	lerp	1	5000	718	713	286	4710	103031.4501	165.0553727	0.0572	0.9428	0.1426	0.1436	21.87504248	0.035043604	432.9339699	43857.32277	44290.25674	8.944326469	0.989768487	1.067523701
co2_peano_no_weekend	69	model3	5	10	5	lerp	1	5000	504	499	215	4780	136943.9135	215.0563152	0.043	0.957	0.0998	0.1008	28.64935428	0.044990861	308.4730433	43876.86487	44185.33791	12.55313179	0.989327659	1.070058554
co2_peano_no_weekend	69	model3	5	15	5	lerp	1	5000	337	332	147	4838	149033.3601	233.3279413	0.0294	0.9706	0.0664	0.0674	30.80474578	0.048228181	207.2706954	43892.36953	44099.64023	18.68234561	0.988978186	1.072137971
co2_peano_no_weekend	69	model3	5	3	7	lerp	1	5000	1669	1664	213	4784	85019.68954	137.2736773	0.0426	0.9574	0.3328	0.3338	17.77167424	0.028694331	860.271839	43778.33144	44638.60328	4.501254825	0.991554374	1.05919306
co2_peano_no_weekend	69	model3	5	5	7	lerp	1	5000	1003	998	248	4747	105912.4318	173.999512	0.0496	0.9504	0.1996	0.2006	22.31144551	0.036654626	555.7817616	43833.97615	44389.75791	6.967308094	0.990295653	1.06513081
co2_peano_no_weekend	69	model3	5	7	7	lerp	1	5000	718	713	224	4772	125445.7799	203.5253065	0.0448	0.9552	0.1426	0.1436	26.28788347	0.042649897	412.865578	43858.57584	44271.44142	9.379088432	0.989740209	1.067977397
co2_peano_no_weekend	69	model3	5	10	7	lerp	1	5000	504	499	167	4828	156961.4128	249.5507216	0.0334	0.9666	0.0998	0.1008	32.51064887	0.051688219	292.9362237	43877.83499	44170.77121	13.21892772	0.989305786	1.070411439
co2_peano_no_weekend	69	model3	5	15	7	lerp	1	5000	337	332	117	4868	173149.0208	273.9222651	0.0234	0.9766	0.0664	0.0674	35.56882103	0.056269981	197.5601832	43892.97586	44090.53604	19.60062348	0.988964525	1.072359355
co2_peano_no_weekend	69	model3	5	3	10	lerp	1	5000	1669	1664	144	4853	114379.1393	184.81443	0.0288	0.9712	0.3328	0.3338	23.56874909	0.038082512	837.9376609	43779.72598	44617.66364	4.621230131	0.991522789	1.059690152
co2_peano_no_weekend	69	model3	5	5	10	lerp	1	5000	1003	998	183	4812	145890.4219	241.0310358	0.0366	0.9634	0.1996	0.2006	30.31804278	0.050089575	534.7423185	43835.28985	44370.03217	7.241436918	0.990265974	1.065604338
co2_peano_no_weekend	69	model3	5	7	10	lerp	1	5000	718	713	168	4828	157921.4254	259.3591338	0.0336	0.9664	0.1426	0.1436	32.70949159	0.053719787	394.7392885	43859.70765	44254.44694	9.809772878	0.989714668	1.068387519
co2_peano_no_weekend	69	model3	5	10	10	lerp	1	5000	504	499	122	4873	197257.5513	321.9862257	0.0244	0.9756	0.0998	0.1008	40.47969449	0.066075564	278.3704554	43878.74447	44157.11493	13.91060974	0.98928528	1.070742481
co2_peano_no_weekend	69	model3	5	15	10	lerp	1	5000	337	332	92	4893	199569.5506	312.5973685	0.0184	0.9816	0.0664	0.0674	40.78674649	0.063886648	189.4680897	43893.48113	44082.94922	20.43775695	0.988953141	1.072543911
co2_peano_no_weekend	69	model3	5	3	25	lerp	1	5000	1669	1664	51	4946	274223.5725	440.8148114	0.0102	0.9898	0.3328	0.3338	55.44350436	0.089125518	807.835073	43781.60559	44589.44066	4.793432343	0.991480222	1.060360885
co2_peano_no_weekend	69	model3	5	5	25	lerp	1	5000	1003	998	80	4915	328918.8967	549.5375847	0.016	0.984	0.1996	0.2006	66.92144389	0.111808257	501.4028931	43837.37157	44338.77446	7.722936623	0.990218949	1.066355562
co2_peano_no_weekend	69	model3	5	7	25	lerp	1	5000	718	713	75	4921	377422.5089	623.7670073	0.015	0.985	0.1426	0.1436	76.69630338	0.126756149	364.6367006	43861.58726	44226.22396	10.61961882	0.989672256	1.069069311
co2_peano_no_weekend	69	model3	5	10	25	lerp	1	5000	504	499	45	4950	356802.0731	565.1530083	0.009	0.991	0.0998	0.1008	72.08122688	0.114172325	253.4468073	43880.30071	44133.74752	15.27856203	0.989250195	1.071309405
co2_peano_no_weekend	69	model3	5	15	25	lerp	1	5000	337	332	42	4943	329024.7599	515.9655585	0.0084	0.9916	0.0664	0.0674	66.56377906	0.104383079	173.2839026	43894.49167	44067.77557	22.346581	0.988930373	1.072913215
pm2p5_peano_no_weekend	69	model3	5	3	1	lerp	1	5000	1669	1664	1471	3526	580.4893733	154.8736935	0.2942	0.7058	0.3328	0.3338	0.164631133	0.043923339	1267.465985	43752.90622	45020.3722	3.055153205	0.992130575	1.050211193
pm2p5_peano_no_weekend	69	model3	5	5	1	lerp	1	5000	1003	998	923	4072	1006.864827	280.3313349	0.1846	0.8154	0.1996	0.2006	0.247265429	0.068843648	774.2682867	43820.33384	44594.60213	5.001241602	0.990603955	1.060238157
pm2p5_peano_no_weekend	69	model3	5	7	1	lerp	1	5000	718	713	670	4326	1274.415932	340.9174965	0.134	0.866	0.1426	0.1436	0.294594529	0.078806633	557.2285264	43849.56181	44406.79034	6.949218467	0.989943667	1.064722274
pm2p5_peano_no_weekend	69	model3	5	10	1	lerp	1	5000	504	499	472	4523	1637.63185	441.1274384	0.0944	0.9056	0.0998	0.1008	0.362067621	0.097529834	391.6597647	43871.67069	44263.33045	9.886904696	0.989444791	1.068173097
pm2p5_peano_no_weekend	69	model3	5	15	1	lerp	1	5000	337	332	319	4666	2266.59127	620.8174654	0.0638	0.9362	0.0664	0.0674	0.485767525	0.133051321	262.9442989	43888.89327	44151.83757	14.72670365	0.989056519	1.070870464
pm2p5_peano_no_weekend	69	model3	5	3	3	lerp	1	5000	1669	1664	1124	3873	622.1162751	163.7265036	0.2248	0.7752	0.3328	0.3338	0.160629041	0.04227382	1155.147727	43759.91937	44915.0671	3.352214332	0.991971572	1.052673453
pm2p5_peano_no_weekend	69	model3	5	5	3	lerp	1	5000	1003	998	766	4229	1033.322938	286.212689	0.1532	0.8468	0.1996	0.2006	0.244342147	0.067678574	723.4499394	43823.50694	44546.95688	5.352551096	0.990532229	1.061372136
pm2p5_peano_no_weekend	69	model3	5	7	3	lerp	1	5000	718	713	562	4434	1289.824884	344.227756	0.1124	0.8876	0.1426	0.1436	0.2908942	0.077633684	522.2706824	43851.74458	44374.01527	7.414359827	0.989894391	1.065508688
pm2p5_peano_no_weekend	69	model3	5	10	3	lerp	1	5000	504	499	407	4588	1649.525887	443.5423382	0.0814	0.9186	0.0998	0.1008	0.35953049	0.096674442	370.6203215	43872.98439	44243.60471	10.44816634	0.989415163	1.068649336
pm2p5_peano_no_weekend	69	model3	5	15	3	lerp	1	5000	337	332	292	4693	2269.514819	621.2563923	0.0584	0.9416	0.0664	0.0674	0.483595742	0.132379372	254.2048379	43889.43896	44143.6438	15.23300185	0.989044222	1.071069235
pm2p5_peano_no_weekend	69	model3	5	3	5	lerp	1	5000	1669	1664	864	4133	708.7009966	184.0829074	0.1728	0.8272	0.3328	0.3338	0.171473747	0.044539779	1070.989954	43765.17419	44836.16414	3.6156294	0.991852468	1.054525954
pm2p5_peano_no_weekend	69	model3	5	5	5	lerp	1	5000	1003	998	620	4375	1080.986985	297.3350037	0.124	0.876	0.1996	0.2006	0.247082739	0.067962287	676.1921132	43826.45772	44502.64983	5.726631072	0.990465538	1.062428843
pm2p5_peano_no_weekend	69	model3	5	7	5	lerp	1	5000	718	713	467	4529	1341.988953	353.6008962	0.0934	0.9066	0.1426	0.1436	0.296310213	0.078074828	491.520727	43853.66461	44345.18534	7.878208493	0.989851051	1.066201402
pm2p5_peano_no_weekend	69	model3	5	10	5	lerp	1	5000	504	499	357	4638	1688.581599	452.6632604	0.0714	0.9286	0.0998	0.1008	0.364075377	0.097598806	354.4361345	43873.99493	44228.43107	10.92524827	0.989392374	1.069015962
pm2p5_peano_no_weekend	69	model3	5	15	5	lerp	1	5000	337	332	275	4710	2277.795884	620.2600853	0.055	0.945	0.0664	0.0674	0.483608468	0.131690039	248.7022143	43889.78255	44138.48476	15.57003735	0.989036479	1.071194424
pm2p5_peano_no_weekend	69	model3	5	3	7	lerp	1	5000	1669	1664	664	4333	815.3696164	209.9176309	0.1328	0.8672	0.3328	0.3338	0.188176694	0.048446257	1006.253206	43769.21635	44775.46956	3.848238935	0.991760868	1.055955398
pm2p5_peano_no_weekend	69	model3	5	5	7	lerp	1	5000	1003	998	480	4515	1170.766215	314.2842519	0.096	0.904	0.1996	0.2006	0.259305917	0.069608915	630.8763895	43829.28724	44460.16363	6.137973826	0.990401596	1.063444102
pm2p5_peano_no_weekend	69	model3	5	7	7	lerp	1	5000	718	713	388	4608	1376.535835	363.8715927	0.0776	0.9224	0.1426	0.1436	0.298727395	0.078965189	465.9497115	43855.26127	44321.21098	8.310559425	0.989815013	1.066778135
pm2p5_peano_no_weekend	69	model3	5	10	7	lerp	1	5000	504	499	317	4678	1704.541182	458.5613906	0.0634	0.9366	0.0998	0.1008	0.364373917	0.098025094	341.4887848	43874.80336	44216.29215	11.33947274	0.989374144	1.069309444
pm2p5_peano_no_weekend	69	model3	5	15	7	lerp	1	5000	337	332	249	4736	2304.440964	630.4416002	0.0498	0.9502	0.0664	0.0674	0.486579595	0.133116892	240.286437	43890.30803	44130.59447	16.11536138	0.989024638	1.071385948
pm2p5_peano_no_weekend	69	model3	5	3	10	lerp	1	5000	1669	1664	470	4527	996.5721148	255.0134379	0.094	0.906	0.3328	0.3338	0.220139632	0.056331663	943.4585605	43773.13725	44716.59581	4.10436974	0.991672033	1.057345666
pm2p5_peano_no_weekend	69	model3	5	5	10	lerp	1	5000	1003	998	376	4619	1262.055696	341.0639331	0.0752	0.9248	0.1996	0.2006	0.27323137	0.073839345	597.2132805	43831.38916	44428.60244	6.483952874	0.990354101	1.064199551
pm2p5_peano_no_weekend	69	model3	5	7	10	lerp	1	5000	718	713	298	4698	1530.494811	394.4709346	0.0596	0.9404	0.1426	0.1436	0.325775822	0.083965716	436.8181748	43857.08024	44293.89842	8.864793156	0.989773961	1.067435933
pm2p5_peano_no_weekend	69	model3	5	10	10	lerp	1	5000	504	499	265	4730	1824.348813	490.8069703	0.053	0.947	0.0998	0.1008	0.385697423	0.103764687	324.6572303	43875.85433	44200.51156	11.92735724	0.989350445	1.069691212
pm2p5_peano_no_weekend	69	model3	5	15	10	lerp	1	5000	337	332	219	4766	2338.400013	637.8298882	0.0438	0.9562	0.0664	0.0674	0.490642051	0.133829183	230.5759248	43890.91435	44121.49028	16.79404634	0.989010975	1.071607021
pm2p5_peano_no_weekend	69	model3	5	3	25	lerp	1	5000	1669	1664	165	4832	2051.07261	508.1720655	0.033	0.967	0.3328	0.3338	0.424476947	0.10516806	844.7350195	43779.30156	44624.03658	4.584044318	0.991532402	1.059538814
pm2p5_peano_no_weekend	69	model3	5	5	25	lerp	1	5000	1003	998	167	4828	2188.318786	576.8468644	0.0334	0.9666	0.1996	0.2006	0.453255755	0.119479467	529.5633786	43835.61323	44365.1766	7.312255573	0.990258669	1.065720964
pm2p5_peano_no_weekend	69	model3	5	7	25	lerp	1	5000	718	713	135	4861	2293.911627	604.1929206	0.027	0.973	0.1426	0.1436	0.471901178	0.124293956	384.0577251	43860.37461	44244.43233	10.08260611	0.989699618	1.068629346
pm2p5_peano_no_weekend	69	model3	5	10	25	lerp	1	5000	504	499	125	4870	2601.937362	683.5959234	0.025	0.975	0.0998	0.1008	0.534278719	0.140368773	279.3415066	43878.68384	44158.02535	13.86225346	0.989286647	1.070720405
pm2p5_peano_no_weekend	69	model3	5	15	25	lerp	1	5000	337	332	118	4867	2947.783743	789.559788	0.0236	0.9764	0.0664	0.0674	0.605667504	0.162227201	197.883867	43892.95565	44090.83951	19.56856224	0.98896498	1.072351974
rad_peano_no_weekend	69	model3	5	3	1	lerp	1	5000	1669	1664	1487	3510	84227.90124	232.8309828	0.2974	0.7026	0.3328	0.3338	23.99655306	0.066333613	1272.644925	43752.58284	45025.22777	3.042720472	0.992137908	1.050097937
rad_peano_no_weekend	69	model3	5	5	1	lerp	1	5000	1003	998	977	4018	169697.6276	522.9943867	0.1954	0.8046	0.1996	0.2006	42.23435231	0.130162864	791.7472087	43819.24246	44610.98967	4.89083223	0.990628627	1.059848686
rad_peano_no_weekend	69	model3	5	7	1	lerp	1	5000	718	713	686	4310	224363.6384	685.6538211	0.1372	0.8628	0.1426	0.1436	52.05652864	0.159084413	562.4074663	43849.23844	44411.64591	6.885226457	0.989950967	1.064605867
rad_peano_no_weekend	69	model3	5	10	1	lerp	1	5000	504	499	473	4522	302003.4168	1008.648422	0.0946	0.9054	0.0998	0.1008	66.78536417	0.223053609	391.9834484	43871.65048	44263.63392	9.878740498	0.989445246	1.068165774
rad_peano_no_weekend	69	model3	5	15	1	lerp	1	5000	337	332	328	4657	349199.1426	1142.504283	0.0656	0.9344	0.0664	0.0674	74.9837111	0.245330531	265.8574525	43888.71137	44154.56883	14.56533465	0.989060618	1.070804223
rad_peano_no_weekend	69	model3	5	3	3	lerp	1	5000	1669	1664	1144	3853	85907.28069	240.8451724	0.2288	0.7712	0.3328	0.3338	22.29620573	0.06250848	1161.621402	43759.51516	44921.13656	3.333532561	0.991980735	1.052531223
rad_peano_no_weekend	69	model3	5	5	3	lerp	1	5000	1003	998	920	4075	170118.0113	524.7982382	0.184	0.816	0.1996	0.2006	41.74675124	0.128784844	773.2972355	43820.39447	44593.69171	5.007521802	0.990602584	1.060259803
rad_peano_no_weekend	69	model3	5	7	3	lerp	1	5000	718	713	612	4384	224686.5634	687.3615901	0.1224	0.8776	0.1426	0.1436	51.25149713	0.156788684	538.4548695	43850.73404	44389.18891	7.191508492	0.989917203	1.065144463
rad_peano_no_weekend	69	model3	5	10	3	lerp	1	5000	504	499	403	4592	302401.0351	1010.953378	0.0806	0.9194	0.0998	0.1008	65.85388395	0.220155352	369.3255866	43873.06523	44242.39082	10.4847942	0.98941334	1.068678656
rad_peano_no_weekend	69	model3	5	15	3	lerp	1	5000	337	332	312	4673	349442.5039	1143.683803	0.0624	0.9376	0.0664	0.0674	74.7790507	0.24474295	260.6785127	43889.03475	44149.71326	14.85470638	0.989053331	1.07092199
rad_peano_no_weekend	69	model3	5	3	5	lerp	1	5000	1669	1664	889	4108	89661.23822	256.4973122	0.1778	0.8222	0.3328	0.3338	21.82600736	0.062438489	1079.082048	43764.66892	44843.75097	3.588515604	0.991863919	1.054347546
rad_peano_no_weekend	69	model3	5	5	5	lerp	1	5000	1003	998	859	4136	171052.9006	530.0441111	0.1718	0.8282	0.1996	0.2006	41.35708428	0.128153799	753.5525273	43821.62733	44575.17986	5.138729718	0.990574715	1.060700123
rad_peano_no_weekend	69	model3	5	7	5	lerp	1	5000	718	713	532	4464	226038.5228	695.3955855	0.1064	0.8936	0.1426	0.1436	50.6358698	0.155778581	512.5601702	43852.35091	44364.91108	7.554825739	0.989880705	1.065727342
rad_peano_no_weekend	69	model3	5	10	5	lerp	1	5000	504	499	353	4642	303482.9216	1017.542163	0.0706	0.9294	0.0998	0.1008	65.37762206	0.219203396	353.1413995	43874.07577	44227.21717	10.9653039	0.989390551	1.069045303
rad_peano_no_weekend	69	model3	5	15	5	lerp	1	5000	337	332	296	4689	349795.7193	1145.830794	0.0592	0.9408	0.0664	0.0674	74.59921503	0.244365706	255.4995728	43889.35812	44144.85769	15.15580916	0.989046044	1.071039782
rad_peano_no_weekend	69	model3	5	3	7	lerp	1	5000	1669	1664	628	4369	96070.42673	283.3783178	0.1256	0.8744	0.3328	0.3338	21.98911118	0.064861139	994.6005915	43769.94394	44764.54454	3.893324415	0.991744382	1.05621311
rad_peano_no_weekend	69	model3	5	5	7	lerp	1	5000	1003	998	803	4192	174563.4245	539.741801	0.1606	0.8394	0.1996	0.2006	41.6420383	0.128755201	735.4262378	43822.75914	44558.18538	5.26538566	0.990549132	1.061104674
rad_peano_no_weekend	69	model3	5	7	7	lerp	1	5000	718	713	458	4538	227815.5533	706.2681486	0.0916	0.9084	0.1426	0.1436	50.20175261	0.155634233	488.6075734	43853.84651	44342.45408	7.925179587	0.989846945	1.066267074
rad_peano_no_weekend	69	model3	5	10	7	lerp	1	5000	504	499	294	4701	305238.6267	1028.291274	0.0588	0.9412	0.0998	0.1008	64.93057365	0.218738837	334.0440588	43875.26821	44209.31227	11.5921917	0.989363662	1.069478269
rad_peano_no_weekend	69	model3	5	15	7	lerp	1	5000	337	332	282	4703	351169.6276	1153.985741	0.0564	0.9436	0.0664	0.0674	74.6692808	0.245372261	250.9680005	43889.64107	44140.60907	15.42946814	0.989039667	1.071142872
rad_peano_no_weekend	69	model3	5	3	10	lerp	1	5000	1669	1664	410	4587	107883.7146	320.2528101	0.082	0.918	0.3328	0.3338	23.51944944	0.069817486	924.037536	43774.3499	44698.38744	4.190633622	0.991644561	1.057776387
rad_peano_no_weekend	69	model3	5	5	10	lerp	1	5000	1003	998	710	4285	182635.9569	568.9988415	0.142	0.858	0.1996	0.2006	42.62216031	0.132788528	705.3236499	43824.63875	44529.9624	5.49010765	0.990506648	1.0617772
rad_peano_no_weekend	69	model3	5	7	10	lerp	1	5000	718	713	368	4628	231426.1622	727.9420292	0.0736	0.9264	0.1426	0.1436	50.00565302	0.157290845	459.4760367	43855.66548	44315.14152	8.427649012	0.98980589	1.066924242
rad_peano_no_weekend	69	model3	5	10	10	lerp	1	5000	504	499	233	4762	308877.5495	1048.225258	0.0466	0.9534	0.0998	0.1008	64.86298813	0.220122902	314.2993506	43876.50107	44190.80042	12.32042879	0.989335862	1.069926282
rad_peano_no_weekend	69	model3	5	15	10	lerp	1	5000	337	332	263	4722	355013.6735	1177.074225	0.0526	0.9474	0.0664	0.0674	75.18290417	0.249274508	244.8180094	43890.02508	44134.84309	15.81706663	0.989031014	1.071282811
rad_peano_no_weekend	69	model3	5	3	25	lerp	1	5000	1669	1664	145	4852	164726.146	514.3879196	0.029	0.971	0.3328	0.3338	33.95015375	0.106015647	838.2613447	43779.70577	44617.96712	4.619445703	0.991523247	1.059682945
rad_peano_no_weekend	69	model3	5	5	25	lerp	1	5000	1003	998	404	4591	241719.4449	856.2751407	0.0808	0.9192	0.1996	0.2006	52.65071769	0.186511684	606.2764252	43830.82326	44437.09969	6.387025135	0.990366888	1.063996055
rad_peano_no_weekend	69	model3	5	7	25	lerp	1	5000	718	713	121	4875	277198.9638	923.278833	0.0242	0.9758	0.1426	0.1436	56.86132591	0.18939053	379.5261527	43860.65756	44240.18371	10.20299323	0.989693234	1.068731972
rad_peano_no_weekend	69	model3	5	10	25	lerp	1	5000	504	499	86	4909	345624.7089	1209.17194	0.0172	0.9828	0.0998	0.1008	70.40633712	0.246317364	266.7178407	43879.47206	44146.18991	14.51834927	0.989268876	1.071007461
rad_peano_no_weekend	69	model3	5	15	25	lerp	1	5000	337	332	161	4824	394118.3166	1438.066681	0.0322	0.9678	0.0664	0.0674	81.6994852	0.298106692	211.8022678	43892.08658	44103.88885	18.28263128	0.988984562	1.072034689
noise_peano_no_weekend	69	model3	5	3	1	lerp	1	5000	1669	1664	1211	3786	190.5009044	98.34221648	0.2422	0.7578	0.3328	0.3338	0.050317196	0.025975229	1183.308212	43758.16103	44941.46924	3.272438005	0.992011433	1.05205503
noise_peano_no_weekend	69	model3	5	5	1	lerp	1	5000	1003	998	855	4140	322.0780855	170.753862	0.171	0.829	0.1996	0.2006	0.077796639	0.041244894	752.2577923	43821.70818	44573.96597	5.147574151	0.990572888	1.06072901
noise_peano_no_weekend	69	model3	5	7	1	lerp	1	5000	718	713	610	4386	425.5150766	217.7299067	0.122	0.878	0.1426	0.1436	0.097016661	0.049642022	537.807502	43850.77446	44388.58197	7.200165026	0.989916291	1.065159027
noise_peano_no_weekend	69	model3	5	10	1	lerp	1	5000	504	499	443	4552	523.5598787	279.6452437	0.0886	0.9114	0.0998	0.1008	0.115017548	0.061433489	382.2729362	43872.2568	44254.52974	10.12968065	0.989431572	1.06838552
noise_peano_no_weekend	69	model3	5	15	1	lerp	1	5000	337	332	300	4685	642.3998971	345.9416827	0.06	0.94	0.0664	0.0674	0.137118441	0.073840274	256.7943078	43889.27728	44146.07158	15.07939487	0.989047866	1.071010332
noise_peano_no_weekend	69	model3	5	3	3	lerp	1	5000	1669	1664	694	4303	235.196593	122.9812642	0.1388	0.8612	0.3328	0.3338	0.054658748	0.028580354	1015.963718	43768.61003	44784.57375	3.811457728	0.991774607	1.055740734
noise_peano_no_weekend	69	model3	5	5	3	lerp	1	5000	1003	998	618	4377	355.805462	188.9414888	0.1236	0.8764	0.1996	0.2006	0.081289802	0.043166893	675.5447458	43826.49814	44502.04289	5.732118843	0.990464624	1.062443333
noise_peano_no_weekend	69	model3	5	7	3	lerp	1	5000	718	713	461	4535	455.8903678	234.5146194	0.0922	0.9078	0.1426	0.1436	0.100527093	0.051712154	489.5786246	43853.78588	44343.3645	7.909460446	0.989848314	1.066245183
noise_peano_no_weekend	69	model3	5	10	3	lerp	1	5000	504	499	360	4635	536.9911613	286.4022015	0.072	0.928	0.0998	0.1008	0.115855698	0.061791198	355.4071857	43873.9343	44229.34148	10.89539807	0.989393742	1.068993957
noise_peano_no_weekend	69	model3	5	15	3	lerp	1	5000	337	332	258	4727	654.5471763	351.7765179	0.0516	0.9484	0.0664	0.0674	0.138469891	0.074418557	243.1995907	43890.12613	44133.32572	15.92232436	0.989028737	1.071319643
noise_peano_no_weekend	69	model3	5	3	5	lerp	1	5000	1669	1664	508	4489	293.215714	154.1121568	0.1016	0.8984	0.3328	0.3338	0.065318716	0.034331066	955.7585426	43772.36924	44728.12779	4.051549208	0.991689432	1.057073057
noise_peano_no_weekend	69	model3	5	5	5	lerp	1	5000	1003	998	476	4519	405.9312057	217.4832051	0.0952	0.9048	0.1996	0.2006	0.089827662	0.048126401	629.5816545	43829.36808	44458.94973	6.150596572	0.990399769	1.063473138
noise_peano_no_weekend	69	model3	5	7	5	lerp	1	5000	718	713	368	4628	514.9898113	265.6290545	0.0736	0.9264	0.1426	0.1436	0.111276969	0.057396079	459.4760367	43855.66548	44315.14152	8.427649012	0.98980589	1.066924242
noise_peano_no_weekend	69	model3	5	10	5	lerp	1	5000	504	499	289	4706	574.7678997	305.6863311	0.0578	0.9422	0.0998	0.1008	0.122135125	0.064956721	332.4256401	43875.36927	44207.79491	11.64862844	0.989361383	1.069514978
noise_peano_no_weekend	69	model3	5	15	5	lerp	1	5000	337	332	211	4774	698.4430887	374.2701591	0.0422	0.9578	0.0664	0.0674	0.146301443	0.078397604	227.9864549	43891.07604	44119.0625	16.98479311	0.989007332	1.07166599
noise_peano_no_weekend	69	model3	5	3	7	lerp	1	5000	1669	1664	407	4590	374.0068505	196.7488704	0.0814	0.9186	0.3328	0.3338	0.081482974	0.042864678	923.0664848	43774.41054	44697.47702	4.195042102	0.991643188	1.057797932
noise_peano_no_weekend	69	model3	5	5	7	lerp	1	5000	1003	998	379	4616	472.1412788	254.4650069	0.0758	0.9242	0.1996	0.2006	0.102283639	0.055126735	598.1843317	43831.32853	44429.51286	6.473427272	0.990355471	1.064177744
noise_peano_no_weekend	69	model3	5	7	7	lerp	1	5000	718	713	301	4695	594.8680154	307.014902	0.0602	0.9398	0.1426	0.1436	0.126702453	0.065391885	437.789226	43857.01961	44294.80883	8.845130341	0.989775329	1.067413993
noise_peano_no_weekend	69	model3	5	10	7	lerp	1	5000	504	499	243	4752	624.4088962	333.7088954	0.0486	0.9514	0.0998	0.1008	0.131399178	0.070224936	317.536188	43876.29897	44193.83515	12.19483924	0.989340419	1.069852811
noise_peano_no_weekend	69	model3	5	15	7	lerp	1	5000	337	332	178	4807	737.9121094	397.1530638	0.0356	0.9644	0.0664	0.0674	0.153507824	0.082619735	217.3048914	43891.743	44109.04789	17.81967604	0.988992304	1.071909303
noise_peano_no_weekend	69	model3	5	3	10	lerp	1	5000	1669	1664	287	4710	516.9215305	272.4854179	0.0574	0.9426	0.3328	0.3338	0.109749794	0.05785253	884.2244359	43776.83584	44661.06027	4.379321142	0.991588249	1.058660464
noise_peano_no_weekend	69	model3	5	5	10	lerp	1	5000	1003	998	288	4707	580.9771928	315.6193581	0.0576	0.9424	0.1996	0.2006	0.123428339	0.067053188	568.7291113	43833.16772	44401.89683	6.808694491	0.990313917	1.064839616
noise_peano_no_weekend	69	model3	5	7	10	lerp	1	5000	718	713	227	4769	735.8795465	381.867828	0.0454	0.9546	0.1426	0.1436	0.154304791	0.080072935	413.8366292	43858.51521	44272.35184	9.357080773	0.989741577	1.067955435
noise_peano_no_weekend	69	model3	5	10	10	lerp	1	5000	504	499	198	4797	732.7240315	392.366262	0.0396	0.9604	0.0998	0.1008	0.152746306	0.081794093	302.9704197	43877.20845	44180.17887	12.78112487	0.989319912	1.070183507
noise_peano_no_weekend	69	model3	5	15	10	lerp	1	5000	337	332	138	4847	835.0229575	450.8505381	0.0276	0.9724	0.0664	0.0674	0.172276245	0.09301641	204.3575418	43892.55143	44096.90897	18.94866582	0.988974088	1.072204376
noise_peano_no_weekend	69	model3	5	3	25	lerp	1	5000	1669	1664	94	4903	1176.984127	630.3113796	0.0188	0.9812	0.3328	0.3338	0.240053871	0.128556268	821.7534739	43780.73652	44602.49	4.712243866	0.991499903	1.060050656
noise_peano_no_weekend	69	model3	5	5	25	lerp	1	5000	1003	998	122	4873	1032.381087	573.3824127	0.0244	0.9756	0.1996	0.2006	0.211857395	0.117665178	514.9976103	43836.52271	44351.52032	7.519069388	0.990238124	1.06604911
noise_peano_no_weekend	69	model3	5	7	25	lerp	1	5000	718	713	79	4917	1405.017726	742.3904748	0.0158	0.9842	0.1426	0.1436	0.285746945	0.150984437	365.9314356	43861.50641	44227.43785	10.58204459	0.98967408	1.069039969
noise_peano_no_weekend	69	model3	5	10	25	lerp	1	5000	504	499	67	4928	1531.31571	798.8096492	0.0134	0.9866	0.0998	0.1008	0.310737766	0.162096114	260.5678496	43879.85607	44140.42392	14.86101517	0.989260219	1.071147365
noise_peano_no_weekend	69	model3	5	15	25	lerp	1	5000	337	332	50	4935	1204.260864	675.7812288	0.01	0.99	0.0664	0.0674	0.244024491	0.136936419	175.8733726	43894.32998	44070.20336	22.01756133	0.988934016	1.07285411
    """,
    header=0,
)
data["error"] *= 0.01

# ### SR vs. error

# +
d = data[["dataset", "time steps", "error", "Suppression Rate"]]
d = d.drop(d[d["error"] == 0.25].index)
d = np.array(np.split(d.values, 4))
co2 = np.array(np.split(d[0], 5))
pm = np.array(np.split(d[1], 5))
rad = np.array(np.split(d[2], 5))
noise = np.array(np.split(d[3], 5))

plot(
    xs=co2[:, :, 1],
    ys=co2[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "co2_SR_error_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="SR",
    grid=True,
)
plot(
    xs=noise[:, :, 1],
    ys=noise[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "noise_SR_error_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="SR",
    grid=True,
)
plot(
    xs=pm[:, :, 1],
    ys=pm[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "pm_SR_error_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="SR",
    grid=True,
)
plot(
    xs=rad[:, :, 1],
    ys=rad[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "rad_SR_error_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="SR",
    grid=True,
)
# -

# ### MAPE vs. error

# +
d = data[["dataset", "time steps", "error", "MAPE"]]
d = d.drop(d[d["error"] == 0.25].index)
d = np.array(np.split(d.values, 4))
co2 = np.array(np.split(d[0], 5))
pm = np.array(np.split(d[1], 5))
rad = np.array(np.split(d[2], 5))
noise = np.array(np.split(d[3], 5))

plot(
    xs=co2[:, :, 1],
    ys=co2[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "co2_MAPE_error_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="MAPE",
    grid=True,
)
plot(
    xs=noise[:, :, 1],
    ys=noise[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "noise_MAPE_error_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="MAPE",
    grid=True,
)
plot(
    xs=pm[:, :, 1],
    ys=pm[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "pm_MAPE_error_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="MAPE",
    grid=True,
)
plot(
    xs=rad[:, :, 1],
    ys=rad[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "rad_MAPE_error_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="$ts$",
    ylabel="MAPE",
    grid=True,
)
# -

# ## Pareto Energy vs. MAPE

# +
d = data[["dataset", "MAPE", "error", "Active Energy [J]"]]
d = d.drop(d[d["error"] == 0.25].index)
d = np.array(np.split(d.values, 4))
co2 = np.array(np.split(d[0], 5))
pm = np.array(np.split(d[1], 5))
rad = np.array(np.split(d[2], 5))
noise = np.array(np.split(d[3], 5))

plot(
    xs=co2[:, :, 1],
    ys=co2[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "co2_pareto_energy_MAPE_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="MAPE",
    ylabel="Active Energy [J]",
    fmts=["o", "x", "s", "^", "+"],
    grid=True,
)
plot(
    xs=noise[:, :, 1],
    ys=noise[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "noise_pareto_energy_MAPE_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="MAPE",
    ylabel="Active Energy [J]",
    fmts=["o", "x", "s", "^", "+"],
    grid=True,
)
plot(
    xs=pm[:, :, 1],
    ys=pm[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "pm_pareto_energy_MAPE_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="MAPE",
    ylabel="Active Energy [J]",
    fmts=["o", "x", "s", "^", "+"],
    grid=True,
)
plot(
    xs=rad[:, :, 1],
    ys=rad[:, :, 3],
    save_path=os.path.join(PLOTS_DIR, "rad_pareto_energy_MAPE_dlds_lerp.pdf"),
    labels=[
        "$\epsilon=0.01$",
        "$\epsilon=0.03$",
        "$\epsilon=0.05$",
        "$\epsilon=0.07$",
        "$\epsilon=0.10$",
    ],
    xlabel="MAPE",
    ylabel="Active Energy [J]",
    fmts=["o", "x", "s", "^", "+"],
    grid=True,
)
# -


