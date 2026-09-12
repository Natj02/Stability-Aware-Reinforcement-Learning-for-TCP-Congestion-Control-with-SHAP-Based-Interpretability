/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Piotr Gawlowicz
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 * Based on script: ./examples/tcp/tcp-variants-comparison.cc
 *
 * Topology:
 *
 *   Right Leafs (Clients)                      Left Leafs (Sinks)
 *           |            \                    /        |
 *           |             \    bottleneck    /         |
 *           |              R0--------------R1          |
 *           |             /                  \         |
 *           |   access   /                    \ access |
 *           N -----------                      --------N
 */

#include "tcp-rl.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/enum.h"
#include "ns3/error-model.h"
#include "ns3/event-id.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/network-module.h"
#include "ns3/opengym-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/traffic-control-module.h"

#include <fstream>
#include <iostream>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpVariantsComparison");

static std::vector<uint32_t> rxPkts;

// ---- baseline telemetry (Sec 3.4.2/3.4.3) ---------------------------------
// Active only when --telemetryOut is set (CUBIC/BBR runs). RL runs leave it
// unset because the gym env logs its own per-interval telemetry. Sampled at
// per-RTT cadence so throughput variance is computed over the same interval
// granularity the RL agent observes.
static std::ofstream g_telem;
static bool g_telemOn = false;
static uint32_t g_telemRunId = 0;
static uint64_t g_rxBytes = 0; // cumulative bytes received at sink
static uint64_t g_lastRxBytes = 0;
static uint32_t g_cwnd = 0;
static double g_rttMs = 0.0;
static uint64_t g_lossCount = 0; // recovery/loss entries since last sample
static uint64_t g_pktsSent = 0;  // cumulative Tx segments
static uint64_t g_lastPktsSent = 0;
static uint32_t g_sampleIdx = 0;
static Time g_sampleIvl = MilliSeconds(100); // updated to current RTT

static void
BaseCwndTrace(uint32_t, uint32_t n)
{
    g_cwnd = n;
}

static void
BaseRttTrace(Time, Time n)
{
    g_rttMs = n.GetMilliSeconds();
    if (n > MilliSeconds(1))
    {
        g_sampleIvl = n; // per-RTT cadence
    }
}

static void
BaseTxTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>)
{
    g_pktsSent++;
}

static void
BaseCongStateTrace(TcpSocketState::TcpCongState_t, TcpSocketState::TcpCongState_t n)
{
    if (n == TcpSocketState::CA_RECOVERY || n == TcpSocketState::CA_LOSS)
    {
        g_lossCount++;
    }
}

static void
BaseSample()
{
    if (!g_telemOn)
    {
        return;
    }
    double ivl = g_sampleIvl.GetSeconds();
    uint64_t dBytes = g_rxBytes - g_lastRxBytes;
    uint64_t dSent = g_pktsSent - g_lastPktsSent;
    g_lastRxBytes = g_rxBytes;
    g_lastPktsSent = g_pktsSent;
    double tputMbps = (ivl > 0) ? (dBytes * 8.0 / ivl / 1e6) : 0.0;

    g_telem << g_telemRunId << "," << g_sampleIdx << "," << ivl << "," << tputMbps << "," << g_rttMs
            << "," << g_cwnd << "," << g_lossCount << "," << dSent << "\n";

    g_sampleIdx++;
    g_lossCount = 0; // per-interval reset
    Simulator::Schedule(g_sampleIvl, &BaseSample);
}

static void
AttachBaselineTracers()
{
    // Sender socket exists once the BulkSend app has started. Wildcard across
    // all nodes; only the left-leaf client runs TCP as a sender here.
    std::string base = "/NodeList/*/$ns3::TcpL4Protocol/SocketList/*/";
    Config::ConnectWithoutContextFailSafe(base + "CongestionWindow", MakeCallback(&BaseCwndTrace));
    Config::ConnectWithoutContextFailSafe(base + "RTT", MakeCallback(&BaseRttTrace));
    Config::ConnectWithoutContextFailSafe(base + "Tx", MakeCallback(&BaseTxTrace));
    Config::ConnectWithoutContextFailSafe(base + "CongState", MakeCallback(&BaseCongStateTrace));
}

static void
CountRxPkts(uint32_t sinkId, Ptr<const Packet> packet, const Address& srcAddr)
{
    rxPkts[sinkId]++;
    g_rxBytes += packet->GetSize(); // for baseline throughput sampling
}

static void
PrintRxCount()
{
    uint32_t size = rxPkts.size();
    NS_LOG_UNCOND("RxPkts:");
    for (uint32_t i = 0; i < size; i++)
    {
        NS_LOG_UNCOND("---SinkId: " << i << " RxPkts: " << rxPkts.at(i));
    }
}

int
main(int argc, char* argv[])
{
    uint32_t openGymPort = 5555;
    double tcpEnvTimeStep = 0.1;

    uint32_t nLeaf = 1;
    std::string transport_prot = "TcpRl";
    double error_p = 0.0;
    std::string bottleneck_bandwidth = "2Mbps";
    std::string bottleneck_delay = "0.01ms";
    std::string access_bandwidth = "10Mbps";
    std::string access_delay = "20ms";
    std::string prefix_file_name = "TcpVariantsComparison";
    uint64_t data_mbytes = 0;
    uint32_t mtu_bytes = 400;
    uint32_t cwndCapSegs = 1500;
    double duration = 10.0;
    uint32_t run = 0;
    uint32_t queueSizePkts = 0; // 0 = use computed BDP (default); >0 = override in packets
    bool flow_monitor = false;
    bool sack = true;
    std::string queue_disc_type = "ns3::PfifoFastQueueDisc";
    std::string recovery = "ns3::TcpClassicRecovery";
    std::string telemetryOut = ""; // baseline telemetry CSV path; empty = off

    CommandLine cmd;
    // required parameters for OpenGym interface
    cmd.AddValue("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
    cmd.AddValue("simSeed", "Seed for random generator. Default: 1", run);
    cmd.AddValue("envTimeStep",
                 "Time step interval for time-based TCP env [s]. Default: 0.1s",
                 tcpEnvTimeStep);
    // other parameters
    cmd.AddValue("nLeaf", "Number of left and right side leaf nodes", nLeaf);
    cmd.AddValue("transport_prot",
                 "Transport protocol to use: TcpNewReno, "
                 "TcpHybla, TcpHighSpeed, TcpHtcp, TcpVegas, TcpScalable, TcpVeno, "
                 "TcpBic, TcpYeah, TcpIllinois, TcpWestwoodPlus, TcpLedbat, "
                 "TcpLp, TcpRl, TcpRlTimeBased, TcpCubic, TcpBbr",
                 transport_prot);
    cmd.AddValue("error_p", "Packet error rate", error_p);
    cmd.AddValue("bottleneck_bandwidth", "Bottleneck bandwidth", bottleneck_bandwidth);
    cmd.AddValue("bottleneck_delay", "Bottleneck delay", bottleneck_delay);
    cmd.AddValue("access_bandwidth", "Access link bandwidth", access_bandwidth);
    cmd.AddValue("access_delay", "Access link delay", access_delay);
    cmd.AddValue("prefix_name", "Prefix of output trace file", prefix_file_name);
    cmd.AddValue("data", "Number of Megabytes of data to transmit", data_mbytes);
    cmd.AddValue("mtu", "Size of IP packets to send in bytes", mtu_bytes);
    cmd.AddValue("cwndCapSegs", "Max congestion window in segments", cwndCapSegs);
    cmd.AddValue("duration", "Time to allow flows to run in seconds", duration);
    cmd.AddValue("run", "Run index (for setting repeatable seeds)", run);
    cmd.AddValue("flow_monitor", "Enable flow monitor", flow_monitor);
    cmd.AddValue("queue_disc_type",
                 "Queue disc type for gateway (e.g. ns3::CoDelQueueDisc)",
                 queue_disc_type);
    cmd.AddValue("queueSize", "Bottleneck queue size in packets (0 = auto BDP)", queueSizePkts);
    cmd.AddValue("sack", "Enable or disable SACK option", sack);
    cmd.AddValue("recovery", "Recovery algorithm type to use (e.g., ns3::TcpPrrRecovery", recovery);
    cmd.AddValue("telemetryOut",
                 "If set, log per-interval baseline telemetry to this CSV path",
                 telemetryOut);
    cmd.Parse(argc, argv);

    transport_prot = std::string("ns3::") + transport_prot;

    SeedManager::SetSeed(1);
    SeedManager::SetRun(run);

    NS_LOG_UNCOND("Ns3Env parameters:");
    if (transport_prot.compare("ns3::TcpRl") == 0 or
        transport_prot.compare("ns3::TcpRlTimeBased") == 0)
    {
        NS_LOG_UNCOND("--openGymPort: " << openGymPort);
    }
    else
    {
        NS_LOG_UNCOND("--openGymPort: No OpenGym");
    }

    NS_LOG_UNCOND("--seed: " << run);
    NS_LOG_UNCOND("--Tcp version: " << transport_prot);

    // OpenGym Env --- has to be created before any other thing
    Ptr<OpenGymInterface> openGymInterface;
    if (transport_prot.compare("ns3::TcpRl") == 0)
    {
        openGymInterface = OpenGymInterface::Get(openGymPort);
        Config::SetDefault("ns3::TcpRl::Reward",
                           DoubleValue(2.0)); // Reward when increasing congestion window
        Config::SetDefault("ns3::TcpRl::Penalty",
                           DoubleValue(-30.0)); // Penalty when decreasing congestion window
    }

    if (transport_prot.compare("ns3::TcpRlTimeBased") == 0)
    {
        openGymInterface = OpenGymInterface::Get(openGymPort);
        Config::SetDefault("ns3::TcpRlTimeBased::StepTime",
                           TimeValue(Seconds(tcpEnvTimeStep))); // Time step of TCP env
        Config::SetDefault("ns3::TcpTimeStepGymEnv::CwndCapSegs", UintegerValue(cwndCapSegs));
    }

    // Calculate the ADU size
    Header* temp_header = new Ipv4Header();
    uint32_t ip_header = temp_header->GetSerializedSize();
    NS_LOG_LOGIC("IP Header size is: " << ip_header);
    delete temp_header;
    temp_header = new TcpHeader();
    uint32_t tcp_header = temp_header->GetSerializedSize();
    NS_LOG_LOGIC("TCP Header size is: " << tcp_header);
    delete temp_header;
    uint32_t tcp_adu_size = mtu_bytes - 20 - (ip_header + tcp_header);
    NS_LOG_LOGIC("TCP ADU size is: " << tcp_adu_size);

    // Set the simulation start and stop time
    double start_time = 0.1;
    double stop_time = start_time + duration;

    // 4 MB of TCP buffer
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 21));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(1 << 21));
    Config::SetDefault("ns3::TcpSocketBase::Sack", BooleanValue(sack));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(2));

    Config::SetDefault("ns3::TcpL4Protocol::RecoveryType",
                       TypeIdValue(TypeId::LookupByName(recovery)));
    // Select TCP variant
    TypeId tcpTid;
    NS_ABORT_MSG_UNLESS(TypeId::LookupByNameFailSafe(transport_prot, &tcpTid),
                        "TypeId " << transport_prot << " not found");
    Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                       TypeIdValue(TypeId::LookupByName(transport_prot)));

    // Create the point-to-point link helpers
    PointToPointHelper bottleNeckLink;
    bottleNeckLink.SetDeviceAttribute("DataRate", StringValue(bottleneck_bandwidth));
    bottleNeckLink.SetChannelAttribute("Delay", StringValue(bottleneck_delay));

    PointToPointHelper pointToPointLeaf;
    pointToPointLeaf.SetDeviceAttribute("DataRate", StringValue(access_bandwidth));
    pointToPointLeaf.SetChannelAttribute("Delay", StringValue(access_delay));

    PointToPointDumbbellHelper d(nLeaf, pointToPointLeaf, nLeaf, pointToPointLeaf, bottleNeckLink);

    // Apply packet loss (Table 3.3.2 loss_rate) on the receiver-side leaf devices.
    // error_p is set from --error_p; 0.0 = no loss.
    if (error_p > 0.0)
    {
        for (uint32_t i = 0; i < d.RightCount(); ++i)
        {
            Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
            em->SetRandomVariable(CreateObject<UniformRandomVariable>());
            em->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
            em->SetRate(error_p);
            d.GetRight(i)->GetDevice(0)->SetAttribute("ReceiveErrorModel", PointerValue(em));
        }
    }

    // Install IP stack
    InternetStackHelper stack;
    stack.InstallAll();

    // Traffic Control
    TrafficControlHelper tchPfifo;
    tchPfifo.SetRootQueueDisc("ns3::PfifoFastQueueDisc");

    TrafficControlHelper tchCoDel;
    tchCoDel.SetRootQueueDisc("ns3::CoDelQueueDisc");

    DataRate access_b(access_bandwidth);
    DataRate bottle_b(bottleneck_bandwidth);
    Time access_d(access_delay);
    Time bottle_d(bottleneck_delay);

    uint32_t size = static_cast<uint32_t>((std::min(access_b, bottle_b).GetBitRate() / 8) *
                                          ((access_d + bottle_d + access_d) * 2).GetSeconds());

    uint32_t qPkts = (queueSizePkts > 0) ? queueSizePkts : (size / mtu_bytes);
    Config::SetDefault("ns3::PfifoFastQueueDisc::MaxSize",
                       QueueSizeValue(QueueSize(QueueSizeUnit::PACKETS, qPkts)));
    Config::SetDefault("ns3::CoDelQueueDisc::MaxSize",
                       QueueSizeValue(QueueSize(QueueSizeUnit::BYTES, size)));

    if (queue_disc_type.compare("ns3::PfifoFastQueueDisc") == 0)
    {
        tchPfifo.Install(d.GetLeft()->GetDevice(1));
        tchPfifo.Install(d.GetRight()->GetDevice(1));
    }
    else if (queue_disc_type.compare("ns3::CoDelQueueDisc") == 0)
    {
        tchCoDel.Install(d.GetLeft()->GetDevice(1));
        tchCoDel.Install(d.GetRight()->GetDevice(1));
    }
    else
    {
        NS_FATAL_ERROR("Queue not recognized. Allowed values are ns3::CoDelQueueDisc or "
                       "ns3::PfifoFastQueueDisc");
    }

    // Assign IP Addresses
    d.AssignIpv4Addresses(Ipv4AddressHelper("10.1.1.0", "255.255.255.0"),
                          Ipv4AddressHelper("10.2.1.0", "255.255.255.0"),
                          Ipv4AddressHelper("10.3.1.0", "255.255.255.0"));

    NS_LOG_INFO("Initialize Global Routing.");
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Install apps in left and right nodes
    uint16_t port = 50000;
    Address sinkLocalAddress(InetSocketAddress(Ipv4Address::GetAny(), port));
    PacketSinkHelper sinkHelper("ns3::TcpSocketFactory", sinkLocalAddress);
    ApplicationContainer sinkApps;
    for (uint32_t i = 0; i < d.RightCount(); ++i)
    {
        sinkHelper.SetAttribute("Protocol", TypeIdValue(TcpSocketFactory::GetTypeId()));
        sinkApps.Add(sinkHelper.Install(d.GetRight(i)));
    }
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(stop_time));

    for (uint32_t i = 0; i < d.LeftCount(); ++i)
    {
        // Create an on/off app sending packets to the left side
        AddressValue remoteAddress(InetSocketAddress(d.GetRightIpv4Address(i), port));
        Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(tcp_adu_size));
        BulkSendHelper ftp("ns3::TcpSocketFactory", Address());
        ftp.SetAttribute("Remote", remoteAddress);
        ftp.SetAttribute("SendSize", UintegerValue(tcp_adu_size));
        ftp.SetAttribute("MaxBytes", UintegerValue(data_mbytes * 1000000));

        ApplicationContainer clientApp = ftp.Install(d.GetLeft(i));
        clientApp.Start(Seconds(start_time * i)); // Start after sink
        clientApp.Stop(Seconds(stop_time - 3));   // Stop before the sink
    }

    // Flow monitor
    FlowMonitorHelper flowHelper;
    if (flow_monitor)
    {
        flowHelper.InstallAll();
    }

    // Count RX packets
    for (uint32_t i = 0; i < d.RightCount(); ++i)
    {
        rxPkts.push_back(0);
        Ptr<PacketSink> pktSink = DynamicCast<PacketSink>(sinkApps.Get(i));
        pktSink->TraceConnectWithoutContext("Rx", MakeBoundCallback(&CountRxPkts, i));
    }

    // Baseline telemetry: attach sender-socket tracers once the app has started
    // (BulkSend creates the socket at start_time), then sample at per-RTT cadence.
    if (!telemetryOut.empty())
    {
        g_telemOn = true;
        g_telemRunId = run;
        g_telem.open(telemetryOut);
        g_telem << "run,interval,interval_s,tput_mbps,rtt_ms,cwnd_b,loss_ct,pkts_sent\n";
        Simulator::Schedule(Seconds(start_time + 0.001), &AttachBaselineTracers);
        Simulator::Schedule(Seconds(start_time + 0.002), &BaseSample);
    }

    Simulator::Stop(Seconds(stop_time));
    Simulator::Run();

    if (flow_monitor)
    {
        flowHelper.SerializeToXmlFile(prefix_file_name + ".flowmonitor", true, true);
    }

    if (transport_prot.compare("ns3::TcpRl") == 0 or
        transport_prot.compare("ns3::TcpRlTimeBased") == 0)
    {
        openGymInterface->NotifySimulationEnd();
    }

    if (g_telemOn)
    {
        g_telem.close();
    }

    PrintRxCount();
    Simulator::Destroy();
    return 0;
}
