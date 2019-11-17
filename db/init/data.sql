--
-- PostgreSQL database dump
--

-- Dumped from database version 10.6
-- Dumped by pg_dump version 10.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: data_db; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE data_db WITH TEMPLATE = template0 ENCODING = 'UTF8';
--LC_COLLATE = 'Russian_Russia.1251' LC_CTYPE = 'Russian_Russia.1251';


ALTER DATABASE data_db OWNER TO postgres;

\connect data_db

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: bg_statistic; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bg_statistic (
    "Id" integer NOT NULL,
    "Battleground" character varying(20),
    "Code" character varying(20),
    "Faction" character varying(20),
    "Class" character varying(20),
    "KB" smallint,
    "D" smallint,
    "HK" smallint,
    "DD" bigint,
    "HD" bigint,
    "Honor" smallint,
    "Win" real,
    "Lose" real,
    "Role" character varying(20),
    "BE" character varying(20)
);


ALTER TABLE public.bg_statistic OWNER TO postgres;

--
-- Name: bgs_data_Id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public."bgs_data_Id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public."bgs_data_Id_seq" OWNER TO postgres;

--
-- Name: bgs_data_Id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public."bgs_data_Id_seq" OWNED BY public.bg_statistic."Id";


--
-- Name: bg_statistic Id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bg_statistic ALTER COLUMN "Id" SET DEFAULT nextval('public."bgs_data_Id_seq"'::regclass);


--
-- Data for Name: bg_statistic; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.bg_statistic ("Id", "Battleground", "Code", "Faction", "Class", "KB", "D", "HK", "DD", "HD", "Honor", "Win", "Lose", "Role", "BE") FROM stdin;
1	WG	WG1	Horde	Hunter	1	3	14	48155	6641	532	1	\N	dps	\N
2	WG	WG1	Horde	Death Knight	1	3	12	27025	7106	377	1	\N	dps	\N
3	WG	WG1	Alliance	Paladin	0	1	19	824	93879	252	\N	1	heal	\N
4	WG	WG1	Alliance	Paladin	1	2	25	7046	98599	274	\N	1	heal	\N
5	WG	WG1	Alliance	Rogue	2	3	23	65483	19629	268	\N	1	dps	\N
6	WG	WG1	Horde	Druid	4	5	12	31759	6071	531	1	\N	dps	\N
7	WG	WG1	Horde	Shaman	0	4	18	12933	94587	541	1	\N	heal	\N
8	WG	WG1	Alliance	Priest	9	5	24	123000	34687	269	\N	1	dps	\N
9	WG	WG1	Horde	Druid	2	1	9	13900	2438	523	1	\N	dps	\N
10	WG	WG1	Alliance	Priest	0	1	25	12566	36734	276	\N	1	heal	\N
11	WG	WG1	Alliance	Rogue	3	1	10	22327	2394	171	\N	1	dps	\N
12	WG	WG1	Alliance	Demon Hunter	0	5	15	29664	4364	242	\N	1	dps	\N
13	WG	WG1	Horde	Paladin	1	1	17	8529	193000	389	1	\N	heal	\N
14	WG	WG1	Horde	Demon Hunter	3	4	18	125000	17460	543	1	\N	dps	\N
15	WG	WG1	Alliance	Warlock	8	2	24	101000	21853	270	\N	1	dps	\N
16	WG	WG1	Alliance	Warlock	5	3	25	124000	27687	274	\N	1	dps	\N
17	WG	WG1	Alliance	Druid	0	0	17	6796	2738	248	\N	1	dps	\N
18	WG	WG1	Horde	Demon Hunter	7	3	16	89349	14996	537	1	\N	dps	\N
19	WG	WG1	Horde	Shaman	1	3	15	43666	17475	383	1	\N	dps	\N
20	WG	WG1	Horde	Warrior	4	1	17	59125	13679	390	1	\N	dps	\N
21	WG	WG2	Alliance	Priest	0	3	17	1632	45269	524	1	\N	heal	\N
22	WG	WG2	Horde	Death Knight	1	2	19	15387	11309	140	\N	1	dps	\N
23	WG	WG2	Alliance	Warrior	2	4	11	20530	1189	509	1	\N	dps	\N
24	WG	WG2	Horde	Death Knight	3	1	21	29076	6734	145	\N	1	dps	\N
25	WG	WG2	Horde	Demon Hunter	6	2	19	43667	6286	137	\N	1	dps	\N
26	WG	WG2	Alliance	Warrior	3	3	16	20454	3284	521	1	\N	dps	\N
27	WG	WG2	Horde	Druid	0	0	0	8162	780	85	\N	1	dps	\N
28	WG	WG2	Alliance	Warlock	6	2	10	25818	11215	523	1	\N	dps	\N
29	WG	WG2	Horde	Shaman	0	4	19	3445	43063	141	\N	1	heal	\N
30	WG	WG2	Horde	Warrior	5	1	20	29930	5117	134	\N	1	dps	\N
31	WG	WG2	Alliance	Death Knight	2	2	17	40511	5841	522	1	\N	dps	\N
32	WG	WG2	Alliance	Monk	0	1	12	8949	15695	510	1	\N	heal	\N
33	WG	WG2	Alliance	Monk	0	0	17	882	9598	747	1	\N	heal	\N
34	WG	WG2	Alliance	Rogue	2	1	18	21731	5777	528	1	\N	dps	\N
35	WG	WG2	Alliance	Demon Hunter	1	0	15	8126	537	742	1	\N	dps	\N
36	WG	WG2	Alliance	Death Knight	3	2	15	41179	15379	740	1	\N	dps	\N
37	WG	WG2	Horde	Demon Hunter	1	3	17	27526	2238	136	\N	1	dps	\N
38	WG	WG2	Horde	Warlock	1	2	19	22769	8918	140	\N	1	dps	\N
39	WG	WG2	Horde	Paladin	0	2	20	14602	6576	143	\N	1	dps	\N
40	WG	WG3	Alliance	Rogue	2	1	28	26266	11364	564	1	\N	dps	\N
41	WG	WG3	Alliance	Paladin	0	0	22	3350	11153	548	1	\N	heal	\N
42	WG	WG3	Alliance	Priest	5	0	30	49500	14576	566	1	\N	dps	\N
43	WG	WG3	Alliance	Druid	0	2	26	23312	12816	781	1	\N	dps	\N
44	WG	WG3	Horde	Paladin	1	4	8	36211	11686	131	\N	1	dps	\N
45	WG	WG3	Alliance	Hunter	8	0	32	89565	4557	575	1	\N	dps	\N
46	WG	WG3	Horde	Druid	1	4	11	28304	9181	141	\N	1	dps	\N
47	WG	WG3	Horde	Shaman	0	6	8	4663	75063	134	\N	1	heal	\N
48	WG	WG3	Alliance	Shaman	7	2	29	70816	7654	789	1	\N	dps	\N
49	WG	WG3	Horde	Druid	3	3	10	55534	11397	136	\N	1	dps	\N
50	WG	WG3	Alliance	Hunter	3	1	30	45806	4899	791	1	\N	dps	\N
51	WG	WG3	Horde	Priest	0	2	5	36350	12216	31	\N	1	dps	\N
52	WG	WG3	Alliance	Priest	0	1	26	20105	157000	553	1	\N	heal	\N
53	WG	WG3	Horde	Rogue	0	2	7	2978	2468	132	\N	1	dps	\N
54	WG	WG3	Alliance	Demon Hunter	5	1	20	24629	6500	476	1	\N	dps	\N
55	WG	WG3	Alliance	Hunter	2	4	22	28534	6625	320	1	\N	dps	\N
56	WG	WG3	Horde	Demon Hunter	1	2	9	19531	2151	134	\N	1	dps	\N
57	WG	WG3	Horde	Mage	1	3	11	76381	18538	139	\N	1	dps	\N
58	WG	WG3	Horde	Demon Hunter	5	6	11	34667	5249	134	\N	1	dps	\N
59	WG	WG4	Horde	Rogue	0	1	11	17990	3102	482	\N	1	dps	1
60	WG	WG4	Horde	Warlock	0	0	6	19587	0	321	\N	1	dps	1
61	WG	WG4	Alliance	Mage	0	2	11	29693	7320	1165	1	\N	dps	1
62	WG	WG4	Horde	Warlock	0	0	9	16048	4390	327	\N	1	dps	1
63	WG	WG4	Alliance	Demon Hunter	2	1	13	14954	437	1171	1	\N	dps	1
64	WG	WG4	Horde	Monk	2	2	11	34637	14788	482	\N	1	dps	1
65	WG	WG4	Horde	Shaman	0	2	11	2329	27087	482	\N	1	heal	1
66	WG	WG4	Alliance	Monk	0	0	11	4040	9774	1165	1	\N	heal	1
67	WG	WG4	Alliance	Demon Hunter	0	1	13	8231	1217	1170	1	\N	dps	1
68	WG	WG4	Horde	Priest	2	4	9	39208	9909	477	\N	1	dps	1
69	WG	WG4	Horde	Warrior	2	0	12	7322	4468	484	\N	1	dps	1
70	WG	WG4	Alliance	Priest	0	0	14	1672	38396	949	1	\N	heal	1
71	WG	WG4	Horde	Priest	3	1	10	22400	48896	479	\N	1	heal	1
72	WG	WG4	Alliance	Monk	0	0	13	1407	65726	945	1	\N	heal	1
73	WG	WG4	Horde	Warrior	1	3	8	27640	1895	325	\N	1	dps	1
74	WG	WG4	Alliance	Warrior	4	3	12	64745	8430	1169	1	\N	dps	1
75	WG	WG4	Alliance	Hunter	1	3	14	34593	5221	1174	1	\N	dps	1
76	WG	WG4	Alliance	Rogue	1	2	12	16745	2911	943	1	\N	dps	1
77	WG	WG4	Horde	Mage	1	1	12	37040	5407	484	\N	1	dps	1
78	WG	WG4	Alliance	Warrior	6	0	13	19212	3305	945	1	\N	dps	1
79	WG	WG5	Horde	Monk	1	0	59	6063	146000	284	\N	1	heal	\N
80	WG	WG5	Horde	Warlock	8	2	54	75901	47521	274	\N	1	dps	\N
81	WG	WG5	Horde	Hunter	10	2	54	78845	10194	274	\N	1	dps	\N
82	WG	WG5	Horde	Demon Hunter	12	2	49	60895	15742	264	\N	1	dps	\N
83	WG	WG5	Alliance	Warrior	3	7	12	75773	20505	518	1	\N	dps	\N
84	WG	WG5	Alliance	Rogue	1	0	15	12761	0	526	1	\N	dps	\N
85	WG	WG5	Alliance	Demon Hunter	5	6	14	88087	12498	540	1	\N	dps	\N
86	WG	WG5	Alliance	Paladin	1	10	9	104000	20599	729	1	\N	dps	\N
87	WG	WG5	Alliance	Paladin	0	5	13	10888	74533	522	1	\N	heal	\N
88	WG	WG5	Alliance	Rogue	0	11	11	63549	9418	740	1	\N	dps	\N
89	WG	WG5	Alliance	Shaman	2	9	6	38201	6893	495	1	\N	dps	\N
90	WG	WG5	Horde	Shaman	1	0	56	11054	121000	279	\N	1	heal	\N
91	WG	WG5	Alliance	Mage	6	5	14	87887	14766	749	1	\N	dps	\N
92	WG	WG5	Alliance	Druid	1	3	15	5682	92506	527	1	\N	heal	\N
93	WG	WG5	Alliance	Rogue	0	5	10	31868	6608	645	1	\N	dps	\N
94	WG	WG5	Horde	Rogue	2	2	46	29559	8989	257	\N	1	dps	\N
95	WG	WG5	Horde	Priest	11	1	57	71710	21055	279	\N	1	dps	\N
96	WG	WG5	Horde	Hunter	4	4	54	88291	5246	274	\N	1	dps	\N
97	WG	WG5	Horde	Monk	4	3	46	60531	23938	257	\N	1	dps	\N
98	WG	WG5	Horde	Warlock	9	3	48	123000	49540	262	\N	1	dps	\N
99	WG	WG6	Horde	Hunter	9	1	54	68673	5739	410	1	\N	dps	\N
100	WG	WG6	Horde	Demon Hunter	7	1	46	87551	12790	543	1	\N	dps	\N
101	WG	WG6	Horde	Priest	10	1	54	85402	18659	411	1	\N	dps	\N
102	WG	WG6	Horde	Shaman	0	3	39	13931	119000	380	1	\N	heal	\N
103	WG	WG6	Alliance	Warlock	0	8	6	31645	15603	233	\N	1	dps	\N
104	WG	WG6	Alliance	Shaman	1	4	10	34909	12436	250	\N	1	dps	\N
105	WG	WG6	Horde	Warrior	7	1	51	56001	9835	551	1	\N	dps	\N
106	WG	WG6	Horde	Mage	11	0	54	93717	11342	410	1	\N	dps	\N
107	WG	WG6	Horde	Druid	4	1	45	30419	22437	392	1	\N	dps	\N
108	WG	WG6	Alliance	Shaman	0	2	0	3698	3062	127	\N	1	dps	\N
109	WG	WG6	Alliance	Priest	2	5	11	30150	120000	252	\N	1	heal	\N
110	WG	WG6	Alliance	Death Knight	1	9	7	44809	18137	236	\N	1	dps	\N
111	WG	WG6	Horde	Monk	0	0	44	4905	89605	389	1	\N	heal	\N
112	WG	WG6	Alliance	Priest	0	7	5	14511	54760	230	\N	1	heal	\N
113	WG	WG6	Alliance	Rogue	0	4	10	55325	9529	249	\N	1	dps	\N
114	WG	WG6	Alliance	Hunter	1	7	6	35374	2135	233	\N	1	dps	\N
115	WG	WG6	Horde	Paladin	4	2	39	59674	14379	530	1	\N	dps	\N
116	WG	WG6	Alliance	Mage	6	5	12	50020	17422	258	\N	1	dps	\N
117	WG	WG6	Horde	Rogue	6	3	43	70482	7562	388	1	\N	dps	\N
118	WG	WG7	Alliance	Priest	0	2	6	3303	52540	259	\N	1	heal	1
119	WG	WG7	Alliance	Paladin	0	4	12	14952	6364	278	\N	1	dps	1
120	WG	WG7	Alliance	Shaman	0	3	18	3565	5069	299	\N	1	heal	1
121	WG	WG7	Alliance	Druid	1	2	18	9863	76747	299	\N	1	heal	1
122	WG	WG7	Horde	Warlock	5	4	24	62078	21283	457	1	\N	dps	1
123	WG	WG7	Horde	Shaman	1	1	26	7517	68328	686	1	\N	heal	1
124	WG	WG7	Alliance	Warrior	3	5	13	47698	4576	281	\N	1	dps	1
125	WG	WG7	Horde	Hunter	6	1	26	55058	9665	461	1	\N	dps	1
126	WG	WG7	Horde	Demon Hunter	2	0	12	12427	10330	660	1	\N	dps	1
127	WG	WG7	Alliance	Mage	2	3	16	51887	10090	290	\N	1	dps	1
128	WG	WG7	Horde	Priest	0	0	13	3095	46782	437	1	\N	heal	1
129	WG	WG7	Alliance	Hunter	0	1	6	26121	0	262	\N	1	dps	1
130	WG	WG7	Horde	Warrior	4	1	23	30117	1747	455	1	\N	dps	1
131	WG	WG7	Alliance	Priest	4	0	16	79295	17671	290	\N	1	dps	1
132	WG	WG7	Horde	Shaman	1	4	25	26991	6369	459	1	\N	dps	1
133	WG	WG7	Horde	Priest	3	3	26	66074	16938	459	1	\N	dps	1
134	WG	WG7	Horde	Rogue	2	2	15	35002	2655	666	1	\N	dps	1
135	WG	WG7	Alliance	Warlock	7	4	17	76383	30058	293	\N	1	dps	1
136	WG	WG7	Horde	Warlock	2	2	26	69414	32434	461	1	\N	dps	1
137	WG	WG7	Alliance	Priest	0	2	14	5967	33584	283	\N	1	heal	1
138	WG	WG8	Horde	Demon Hunter	1	2	11	36661	24957	652	1	\N	dps	1
139	WG	WG8	Alliance	Warrior	4	2	14	51033	5913	338	\N	1	dps	1
140	WG	WG8	Alliance	Druid	0	6	4	41339	2763	299	\N	1	dps	1
141	WG	WG8	Horde	Priest	0	0	10	5543	44516	418	1	\N	heal	1
142	WG	WG8	Horde	Druid	3	0	18	46056	10488	670	1	\N	dps	1
143	WG	WG8	Alliance	Mage	0	3	12	29346	5065	329	\N	1	dps	1
144	WG	WG8	Alliance	Warlock	3	2	11	110000	19493	323	\N	1	dps	1
145	WG	WG8	Horde	Death Knight	1	5	17	44465	20298	668	1	\N	dps	1
146	WG	WG8	Horde	Shaman	2	2	20	14838	81762	671	1	\N	heal	1
147	WG	WG8	Alliance	Monk	0	0	12	13460	79489	332	\N	1	heal	1
148	WG	WG8	Horde	Mage	7	2	20	74679	11349	671	1	\N	dps	1
149	WG	WG8	Horde	Druid	0	1	17	5272	69559	667	1	\N	heal	1
150	WG	WG8	Alliance	Mage	1	1	11	39616	11059	323	\N	1	dps	1
151	WG	WG8	Alliance	Hunter	4	2	12	63412	4181	326	\N	1	dps	1
152	WG	WG8	Alliance	Paladin	1	3	8	47194	16757	314	\N	1	dps	1
153	WG	WG8	Horde	Druid	0	1	12	21719	24253	429	1	\N	heal	1
154	WG	WG8	Alliance	Priest	1	1	13	30487	50333	332	\N	1	heal	1
155	WG	WG8	Alliance	Warrior	1	3	4	16274	36748	304	\N	1	dps	1
156	WG	WG8	Horde	Death Knight	7	0	18	49866	45850	444	1	\N	dps	1
157	WG	WG8	Horde	Death Knight	1	1	15	44877	46855	439	1	\N	dps	1
158	WG	WG9	Horde	Priest	7	0	26	50293	3258	505	1	\N	dps	\N
159	WG	WG9	Horde	Druid	0	0	15	22464	3277	487	1	\N	dps	\N
160	WG	WG9	Alliance	Priest	0	1	4	6069	47212	156	\N	1	heal	\N
161	WG	WG9	Horde	Warrior	4	1	25	18305	5869	507	1	\N	dps	\N
162	WG	WG9	Alliance	Mage	0	5	2	19958	10411	149	\N	1	dps	\N
163	WG	WG9	Horde	Paladin	2	0	26	30930	2752	509	1	\N	dps	\N
164	WG	WG9	Horde	Death Knight	3	1	23	44733	9903	503	1	\N	dps	\N
165	WG	WG9	Horde	Shaman	1	0	26	9903	46222	509	1	\N	heal	\N
166	WG	WG9	Alliance	Mage	0	0	2	11558	2538	149	\N	1	dps	\N
167	WG	WG9	Horde	Rogue	4	0	25	33904	1517	507	1	\N	dps	\N
168	WG	WG9	Alliance	Death Knight	2	3	4	31903	8838	161	\N	1	dps	\N
169	WG	WG9	Alliance	Warrior	0	3	2	10869	164	151	\N	1	dps	\N
170	WG	WG9	Horde	Shaman	0	0	24	2586	13680	505	1	\N	heal	\N
171	WG	WG9	Horde	Rogue	0	3	17	31081	8095	491	1	\N	dps	\N
172	WG	WG9	Horde	Shaman	1	1	21	19440	165	499	1	\N	dps	\N
173	WG	WG9	Alliance	Death Knight	2	2	4	23606	15150	163	\N	1	dps	\N
174	WG	WG9	Alliance	Warrior	2	4	3	24925	5477	152	\N	1	dps	\N
175	WG	WG9	Alliance	Monk	0	4	4	520	27241	156	\N	1	heal	\N
176	WG	WG10	Alliance	Death Knight	4	3	29	104000	17438	588	1	\N	dps	\N
177	WG	WG10	Alliance	Rogue	4	1	26	36320	9261	578	1	\N	dps	\N
178	WG	WG10	Alliance	Priest	2	5	25	12103	68259	795	1	\N	heal	\N
179	WG	WG10	Horde	Rogue	0	2	23	13909	9234	166	\N	1	dps	\N
180	WG	WG10	Alliance	Priest	7	4	28	73185	22188	570	1	\N	dps	\N
181	WG	WG10	Horde	Monk	6	3	27	68056	27223	174	\N	1	dps	\N
182	WG	WG10	Horde	Shaman	1	4	23	10614	119000	165	\N	1	heal	\N
183	WG	WG10	Alliance	Shaman	6	5	30	93721	12013	813	1	\N	dps	\N
184	WG	WG10	Alliance	Mage	1	3	30	49899	10739	591	1	\N	dps	\N
185	WG	WG10	Alliance	Death Knight	1	0	18	12952	7036	782	1	\N	dps	\N
186	WG	WG10	Horde	Warrior	5	3	23	55932	17877	166	\N	1	dps	\N
187	WG	WG10	Horde	Shaman	5	3	26	70027	5397	171	\N	1	dps	\N
188	WG	WG10	Horde	Priest	0	3	26	272	79726	172	\N	1	heal	\N
189	WG	WG10	Horde	Hunter	4	4	23	83092	9739	165	\N	1	dps	\N
190	WG	WG10	Horde	Warrior	0	1	10	8670	161	124	\N	1	dps	\N
191	WG	WG10	Horde	Warrior	1	0	10	12331	9887	114	\N	1	dps	\N
192	WG	WG10	Horde	Shaman	0	2	10	0	34461	124	\N	1	heal	\N
193	WG	WG10	Alliance	Death Knight	6	1	27	72057	15091	577	1	\N	dps	\N
194	WG	WG10	Alliance	Warlock	2	3	29	44982	23095	254	1	\N	dps	\N
195	WG	WG10	Alliance	Priest	0	2	28	3011	60718	583	1	\N	heal	\N
196	WG	WG11	Horde	Hunter	0	9	7	12690	3914	199	\N	1	dps	\N
197	WG	WG11	Horde	Hunter	1	6	9	24268	4855	202	\N	1	dps	\N
198	WG	WG11	Alliance	Druid	0	0	45	4719	39014	550	1	\N	heal	\N
199	WG	WG11	Alliance	Warlock	9	1	46	65209	24585	555	1	\N	dps	\N
200	WG	WG11	Horde	Death Knight	0	5	9	32829	5580	202	\N	1	dps	\N
201	WG	WG11	Alliance	Hunter	4	2	45	19330	2813	550	1	\N	dps	\N
202	WG	WG11	Horde	Rogue	2	5	12	29343	3125	137	\N	1	dps	\N
203	WG	WG11	Horde	Druid	0	2	7	2637	27010	198	\N	1	heal	\N
204	WG	WG11	Horde	Shaman	0	8	10	4867	53987	206	\N	1	heal	\N
205	WG	WG11	Alliance	Shaman	4	1	41	36457	17876	543	1	\N	dps	\N
206	WG	WG11	Horde	Mage	1	5	11	25866	9945	133	\N	1	dps	\N
207	WG	WG11	Horde	Warlock	1	9	8	29725	24875	200	\N	1	dps	\N
208	WG	WG11	Alliance	Warlock	9	2	44	40436	14150	400	1	\N	dps	\N
209	WG	WG11	Horde	Mage	3	4	12	23596	5084	211	\N	1	dps	\N
210	WG	WG11	Alliance	Druid	0	1	25	839	20727	493	1	\N	heal	\N
211	WG	WG11	Alliance	Monk	9	2	48	42353	39026	408	1	\N	dps	\N
212	WG	WG11	Alliance	Rogue	2	0	39	26556	4351	388	1	\N	dps	\N
213	WG	WG11	Alliance	Druid	15	3	48	82954	4568	557	1	\N	dps	\N
214	WG	WG11	Alliance	Paladin	2	0	44	1385	30214	899	1	\N	heal	\N
215	WG	WG11	Horde	Druid	5	3	10	40427	11455	208	\N	1	dps	\N
216	WG	WG12	Alliance	Shaman	6	7	10	31303	3313	143	\N	1	dps	\N
217	WG	WG12	Alliance	Priest	0	5	12	4802	15801	147	\N	1	heal	\N
218	WG	WG12	Alliance	Rogue	1	5	10	11574	8768	151	\N	1	dps	\N
219	WG	WG12	Horde	Mage	5	2	40	25577	3771	405	1	\N	dps	\N
220	WG	WG12	Horde	Death Knight	10	2	46	30203	11833	413	1	\N	dps	\N
221	WG	WG12	Horde	Warrior	6	2	37	28421	6228	403	1	\N	dps	\N
222	WG	WG12	Horde	Mage	7	1	48	32630	7234	421	1	\N	dps	\N
223	WG	WG12	Horde	Mage	8	1	47	37857	8706	415	1	\N	dps	\N
224	WG	WG12	Horde	Shaman	1	2	46	3663	55865	563	1	\N	heal	\N
225	WG	WG12	Alliance	Rogue	1	6	8	11267	2047	139	\N	1	dps	\N
226	WG	WG12	Alliance	Paladin	0	3	12	16628	5412	148	\N	1	dps	\N
227	WG	WG12	Horde	Druid	2	1	42	27636	8714	405	1	\N	dps	\N
228	WG	WG12	Horde	Rogue	6	1	44	15198	1431	409	1	\N	dps	\N
229	WG	WG12	Alliance	Warlock	3	8	13	14011	12435	157	\N	1	dps	\N
230	WG	WG12	Alliance	Hunter	2	6	12	44476	7369	147	\N	1	dps	\N
231	WG	WG12	Horde	Warrior	6	3	39	16179	6773	561	1	\N	dps	\N
232	WG	WG12	Horde	Warlock	0	1	50	6225	5529	573	1	\N	dps	\N
233	WG	WG12	Alliance	Priest	1	4	12	15404	2489	151	\N	1	dps	\N
234	WG	WG12	Alliance	Druid	1	3	10	11126	7020	146	\N	1	dps	\N
235	WG	WG12	Alliance	Rogue	0	2	2	3434	308	25	\N	1	dps	\N
236	WG	WG13	Horde	Warlock	2	3	34	18854	8143	547	1	\N	dps	\N
237	WG	WG13	Alliance	Paladin	1	5	7	32645	15990	203	\N	1	dps	\N
238	WG	WG13	Alliance	Druid	0	1	11	2396	5822	217	\N	1	heal	\N
239	WG	WG13	Horde	Demon Hunter	7	1	29	41082	3627	540	1	\N	dps	\N
240	WG	WG13	Horde	Demon Hunter	4	1	36	50216	6136	551	1	\N	dps	\N
241	WG	WG13	Horde	Hunter	7	1	36	27115	2705	404	1	\N	dps	\N
242	WG	WG13	Alliance	Hunter	4	5	11	51078	8346	218	\N	1	dps	\N
243	WG	WG13	Horde	Shaman	0	0	36	7887	87513	554	1	\N	heal	\N
244	WG	WG13	Alliance	Mage	1	3	11	27926	5377	216	\N	1	dps	\N
245	WG	WG13	Alliance	Paladin	0	3	9	3609	10401	214	\N	1	heal	\N
246	WG	WG13	Horde	Hunter	3	3	34	52228	6371	397	1	\N	dps	\N
247	WG	WG13	Horde	Monk	1	0	22	15311	12046	522	1	\N	dps	\N
248	WG	WG13	Alliance	Warrior	0	7	10	55846	17201	213	\N	1	dps	\N
249	WG	WG13	Horde	Priest	7	0	36	38715	78382	401	1	\N	heal	\N
250	WG	WG13	Alliance	Druid	2	2	11	35735	15773	220	\N	1	dps	\N
251	WG	WG13	Horde	Warrior	1	3	23	17630	1657	524	1	\N	dps	\N
252	WG	WG13	Horde	Paladin	4	2	32	2004	16159	543	1	\N	heal	\N
253	WG	WG13	Alliance	Monk	1	2	7	11591	5171	197	\N	1	dps	\N
254	WG	WG13	Alliance	Demon Hunter	2	4	10	24451	1763	215	\N	1	dps	\N
255	WG	WG13	Alliance	Warrior	3	5	11	34120	9819	218	\N	1	dps	\N
256	WG	WG14	Alliance	Shaman	1	3	40	3053	44130	327	1	\N	heal	\N
257	WG	WG14	Horde	Demon Hunter	9	5	33	49080	2493	193	\N	1	dps	\N
258	WG	WG14	Horde	Warrior	7	5	26	33087	5365	178	\N	1	dps	\N
259	WG	WG14	Alliance	Rogue	6	2	47	16415	3386	342	1	\N	dps	\N
260	WG	WG14	Horde	Shaman	0	4	25	7472	48106	177	\N	1	heal	\N
261	WG	WG14	Alliance	Druid	0	4	45	778	34434	487	1	\N	heal	\N
262	WG	WG14	Alliance	Demon Hunter	4	3	45	32057	1696	487	1	\N	dps	\N
263	WG	WG14	Horde	Warrior	4	7	32	31141	1842	189	\N	1	dps	\N
264	WG	WG14	Alliance	Demon Hunter	3	4	38	37113	5305	473	1	\N	dps	\N
265	WG	WG14	Horde	Mage	2	4	21	29112	7219	168	\N	1	dps	\N
266	WG	WG14	Alliance	Monk	1	4	33	14806	54073	313	1	\N	heal	\N
267	WG	WG14	Alliance	Druid	6	5	44	24422	7954	486	1	\N	dps	\N
268	WG	WG14	Horde	Warrior	2	4	14	7757	6224	153	\N	1	dps	\N
269	WG	WG14	Alliance	Hunter	12	3	50	31056	6336	503	1	\N	dps	\N
270	WG	WG14	Alliance	Warlock	13	5	51	43470	16041	505	1	\N	dps	\N
271	WG	WG14	Horde	Warrior	1	7	25	18769	1247	173	\N	1	dps	\N
272	WG	WG14	Horde	Druid	6	7	28	35975	9485	182	\N	1	dps	\N
273	WG	WG14	Horde	Monk	2	5	29	38337	9363	182	\N	1	dps	\N
274	WG	WG14	Alliance	Death Knight	7	3	51	23094	5677	505	1	\N	dps	\N
275	WG	WG14	Horde	Warrior	3	7	23	20776	9371	171	\N	1	dps	\N
276	WG	WG15	Alliance	Demon Hunter	0	7	1	17372	3038	107	\N	1	dps	\N
277	WG	WG15	Horde	Rogue	3	1	32	9029	397	365	1	\N	dps	\N
278	WG	WG15	Horde	Druid	6	0	33	24448	4582	367	1	\N	dps	\N
279	WG	WG15	Horde	Hunter	9	0	35	19884	3425	371	1	\N	dps	\N
280	WG	WG15	Horde	Paladin	0	0	26	2599	1641	353	1	\N	dps	\N
281	WG	WG15	Horde	Shaman	0	0	27	1954	15099	505	1	\N	heal	\N
282	WG	WG15	Horde	Warrior	3	0	36	11996	1038	523	1	\N	dps	\N
283	WG	WG15	Alliance	Demon Hunter	1	2	1	10741	1466	107	\N	1	dps	\N
284	WG	WG15	Alliance	Priest	0	3	1	955	8527	107	\N	1	heal	\N
285	WG	WG15	Alliance	Rogue	0	3	1	9862	211	107	\N	1	dps	\N
286	WG	WG15	Horde	Warrior	6	0	36	18433	2831	373	1	\N	dps	\N
287	WG	WG15	Horde	Druid	0	0	35	911	26698	521	1	\N	heal	\N
288	WG	WG15	Alliance	Warrior	0	4	1	3043	2887	107	\N	1	dps	\N
289	WG	WG15	Alliance	Hunter	0	5	1	7294	3281	107	\N	1	dps	\N
290	WG	WG15	Horde	Paladin	4	0	36	14970	3268	373	1	\N	dps	\N
291	WG	WG15	Alliance	Druid	0	2	0	0	10195	95	\N	1	heal	\N
292	WG	WG15	Alliance	Warlock	0	4	0	16450	7277	105	\N	1	dps	\N
293	WG	WG15	Horde	Monk	6	0	34	23189	6444	369	1	\N	dps	\N
294	WG	WG15	Alliance	Rogue	0	5	1	6858	2459	107	\N	1	dps	\N
295	WG	WG16	Alliance	Priest	10	4	58	40092	14538	419	1	\N	dps	\N
296	WG	WG16	Alliance	Death Knight	12	6	53	31347	14346	410	1	\N	dps	\N
297	WG	WG16	Alliance	Druid	9	2	62	70441	494	430	1	\N	dps	\N
298	WG	WG16	Horde	Priest	2	4	22	17089	15538	228	\N	1	dps	\N
299	WG	WG16	Alliance	Mage	4	1	61	35490	3726	425	1	\N	dps	\N
300	WG	WG16	Horde	Demon Hunter	1	7	25	29034	1174	234	\N	1	dps	\N
301	WG	WG16	Horde	Warrior	5	4	25	19992	3683	233	\N	1	dps	\N
302	WG	WG16	Horde	Paladin	3	11	21	24742	11005	230	\N	1	dps	\N
303	WG	WG16	Horde	Shaman	0	10	25	4955	43833	234	\N	1	heal	\N
304	WG	WG16	Alliance	Hunter	2	5	57	31076	4839	416	1	\N	dps	\N
305	WG	WG16	Alliance	Warrior	2	2	54	18823	5387	561	1	\N	dps	\N
306	WG	WG16	Horde	Paladin	0	6	27	2618	57836	242	\N	1	heal	\N
307	WG	WG16	Alliance	Paladin	13	3	57	37267	11643	422	1	\N	dps	\N
308	WG	WG16	Alliance	Druid	3	3	56	17418	16583	416	1	\N	dps	\N
309	WG	WG16	Horde	Monk	2	5	25	31043	16953	238	\N	1	dps	\N
310	WG	WG16	Alliance	Priest	13	3	62	48575	19527	578	1	\N	dps	\N
311	WG	WG16	Horde	Paladin	8	9	23	42133	9465	230	\N	1	dps	\N
312	WG	WG16	Horde	Mage	2	4	24	25663	13130	233	\N	1	dps	\N
313	WG	WG16	Horde	Paladin	7	9	23	32757	12189	230	\N	1	dps	\N
314	WG	WG16	Alliance	Shaman	1	3	61	6332	54914	580	1	\N	heal	\N
315	WG	WG17	Alliance	Warlock	3	2	23	30895	7157	665	1	\N	dps	\N
316	WG	WG17	Horde	Mage	8	6	38	85175	23305	219	\N	1	dps	\N
317	WG	WG17	Horde	Warrior	9	6	40	46851	8477	225	\N	1	dps	\N
318	WG	WG17	Alliance	Hunter	2	5	46	38946	11987	782	1	\N	dps	\N
319	WG	WG17	Alliance	Priest	15	4	49	123000	35145	563	1	\N	dps	\N
320	WG	WG17	Alliance	Hunter	7	8	41	132000	25295	532	1	\N	dps	\N
321	WG	WG17	Horde	Shaman	0	6	40	10874	150000	225	\N	1	heal	\N
322	WG	WG17	Horde	Warlock	3	5	22	53580	38579	189	\N	1	dps	\N
323	WG	WG17	Alliance	Demon Hunter	4	4	37	41529	9268	747	1	\N	dps	\N
324	WG	WG17	Horde	Warlock	3	5	31	19953	17449	208	\N	1	dps	\N
325	WG	WG17	Horde	Death Knight	2	6	41	42427	26412	226	\N	1	dps	\N
326	WG	WG17	Alliance	Warlock	3	5	27	52437	35221	718	1	\N	dps	\N
327	WG	WG17	Horde	Hunter	8	5	47	47375	13113	239	\N	1	dps	\N
328	WG	WG17	Horde	Paladin	7	4	35	50943	27311	213	\N	1	dps	\N
329	WG	WG17	Alliance	Druid	0	6	47	1244	81587	779	1	\N	heal	\N
330	WG	WG17	Alliance	Druid	5	3	43	89434	8523	771	1	\N	dps	\N
331	WG	WG17	Alliance	Hunter	7	3	47	63903	12224	562	1	\N	dps	\N
332	WG	WG17	Horde	Warrior	5	8	39	59106	23215	222	\N	1	dps	\N
333	WG	WG17	Horde	Paladin	6	6	42	78565	42581	228	\N	1	dps	\N
334	WG	WG17	Alliance	Demon Hunter	11	11	48	109000	9050	562	1	\N	dps	\N
335	BG	BG1	Alliance	Monk	0	2	17	1404	111000	213	\N	1	heal	\N
336	BG	BG1	Horde	Shaman	2	4	17	23373	5920	358	1	\N	dps	\N
337	BG	BG1	Alliance	Paladin	3	4	17	72205	35392	211	\N	1	dps	\N
338	BG	BG1	Alliance	Demon Hunter	0	6	1	13751	1416	177	\N	1	dps	\N
339	BG	BG1	Horde	Warlock	3	1	25	46804	16934	520	1	\N	dps	\N
340	BG	BG1	Horde	Mage	4	1	16	39910	3120	364	1	\N	dps	\N
341	BG	BG1	Horde	Warrior	5	1	27	62571	17566	374	1	\N	dps	\N
342	BG	BG1	Alliance	Priest	1	3	19	7340	48071	216	\N	1	heal	\N
343	BG	BG1	Alliance	Hunter	3	3	19	46793	11099	218	\N	1	dps	\N
344	BG	BG1	Alliance	Shaman	0	3	24	14956	70297	232	\N	1	heal	\N
345	BG	BG1	Horde	Shaman	0	8	19	3896	70444	508	1	\N	heal	\N
346	BG	BG1	Horde	Warrior	4	0	30	38997	9283	384	1	\N	dps	\N
347	BG	BG1	Alliance	Mage	1	6	11	20832	8656	197	\N	1	dps	\N
348	BG	BG1	Horde	Warlock	4	2	27	79679	31249	374	1	\N	dps	\N
349	BG	BG1	Alliance	Monk	2	2	21	34753	11306	224	\N	1	dps	\N
350	BG	BG1	Alliance	Warrior	13	2	24	99446	16581	232	\N	1	dps	\N
351	BG	BG1	Horde	Priest	1	3	18	16276	79725	514	1	\N	heal	\N
352	BG	BG1	Horde	Hunter	6	2	24	57857	17782	368	1	\N	dps	\N
353	BG	BG1	Horde	Warrior	0	3	16	16256	3467	352	1	\N	dps	\N
354	BG	BG1	Alliance	Hunter	0	1	7	9475	1849	189	\N	1	dps	\N
355	BG	BG2	Alliance	Paladin	1	4	11	12144	4054	188	\N	1	dps	\N
356	BG	BG2	Alliance	Demon Hunter	0	7	9	12337	6720	183	\N	1	dps	\N
357	BG	BG2	Alliance	Demon Hunter	3	6	13	36969	3986	194	\N	1	dps	\N
358	BG	BG2	Alliance	Warrior	0	6	8	14450	3295	184	\N	1	dps	\N
359	BG	BG2	Horde	Shaman	1	2	45	5337	55768	547	1	\N	heal	\N
360	BG	BG2	Alliance	Druid	0	5	7	270	28254	183	\N	1	heal	\N
361	BG	BG2	Alliance	Demon Hunter	1	0	4	8489	309	173	\N	1	dps	\N
362	BG	BG2	Horde	Warlock	5	1	48	16152	8432	405	1	\N	dps	\N
363	BG	BG2	Horde	Warlock	4	2	52	31053	12454	411	1	\N	dps	\N
364	BG	BG2	Horde	Priest	0	2	46	1484	39236	399	1	\N	heal	\N
365	BG	BG2	Horde	Mage	3	1	49	31740	3400	405	1	\N	dps	\N
366	BG	BG2	Horde	Hunter	11	0	54	35220	887	415	1	\N	dps	\N
367	BG	BG2	Horde	Shaman	6	1	24	13117	1164	358	1	\N	dps	\N
368	BG	BG2	Alliance	Warrior	2	5	9	17113	1766	187	\N	1	dps	\N
369	BG	BG2	Alliance	Paladin	3	6	9	31140	8288	187	\N	1	dps	\N
370	BG	BG2	Alliance	Paladin	4	8	10	49434	18988	185	\N	1	dps	\N
371	BG	BG2	Horde	Mage	12	1	56	54907	3873	419	1	\N	dps	\N
372	BG	BG2	Horde	Death Knight	7	2	51	33027	17150	409	1	\N	dps	\N
373	BG	BG2	Horde	Mage	6	2	51	38638	2456	559	1	\N	dps	\N
374	BG	BG2	Alliance	Shaman	0	9	10	6983	41303	186	\N	1	heal	\N
375	BG	BG3	Alliance	Monk	0	6	4	9076	8398	143	\N	1	dps	\N
376	BG	BG3	Horde	Death Knight	5	0	22	25425	6302	520	1	\N	dps	\N
377	BG	BG3	Horde	Hunter	4	1	29	30894	2304	377	1	\N	dps	\N
378	BG	BG3	Horde	Druid	0	0	37	5414	70045	542	1	\N	heal	\N
379	BG	BG3	Horde	Warlock	6	1	24	25013	16005	367	1	\N	dps	\N
380	BG	BG3	Alliance	Demon Hunter	0	6	6	8472	1798	147	\N	1	dps	\N
381	BG	BG3	Horde	Paladin	1	1	34	10538	24935	544	1	\N	heal	\N
382	BG	BG3	Alliance	Monk	0	5	8	1556	44968	152	\N	1	heal	\N
383	BG	BG3	Alliance	Druid	0	3	5	17832	2562	146	\N	1	dps	\N
384	BG	BG3	Alliance	Shaman	0	0	3	7081	442	101	\N	1	dps	\N
385	BG	BG3	Horde	Rogue	2	1	12	11182	1967	343	1	\N	dps	\N
386	BG	BG3	Horde	Rogue	3	1	24	16420	8102	367	1	\N	dps	\N
387	BG	BG3	Alliance	Demon Hunter	3	4	4	38595	6677	144	\N	1	dps	\N
388	BG	BG3	Alliance	Warrior	1	1	5	6260	870	145	\N	1	dps	\N
389	BG	BG3	Horde	Shaman	0	1	24	3903	21364	516	1	\N	heal	\N
390	BG	BG3	Horde	Hunter	9	0	33	49713	5274	534	1	\N	dps	\N
391	BG	BG3	Alliance	Priest	0	5	3	5137	29520	141	\N	1	heal	\N
392	BG	BG3	Alliance	Warrior	1	5	4	12083	91	143	\N	1	dps	\N
393	BG	BG3	Alliance	Mage	3	3	7	28483	5374	65	\N	1	dps	\N
394	BG	BG3	Horde	Shaman	11	2	38	38895	6568	544	1	\N	dps	\N
395	BG	BG4	Alliance	Shaman	0	2	19	5362	1982	764	1	\N	dps	\N
396	BG	BG4	Alliance	Priest	9	0	37	68070	7779	604	1	\N	dps	\N
397	BG	BG4	Alliance	Monk	7	1	37	33215	7643	604	1	\N	dps	\N
398	BG	BG4	Alliance	Paladin	8	0	39	72723	22079	612	1	\N	dps	\N
399	BG	BG4	Horde	Hunter	0	4	0	35874	5516	115	\N	1	dps	\N
400	BG	BG4	Horde	Hunter	0	3	2	8092	4200	110	\N	1	dps	\N
401	BG	BG4	Horde	Mage	3	2	5	16626	3718	126	\N	1	dps	\N
402	BG	BG4	Horde	Shaman	0	9	0	3159	31023	115	\N	1	heal	\N
403	BG	BG4	Horde	Monk	0	3	1	10158	15965	117	\N	1	heal	\N
404	BG	BG4	Alliance	Demon Hunter	1	0	9	9851	1973	509	1	\N	dps	\N
405	BG	BG4	Horde	Warlock	1	1	5	38656	17995	136	\N	1	dps	\N
406	BG	BG4	Horde	Mage	1	3	3	14831	13896	124	\N	1	dps	\N
407	BG	BG4	Horde	Paladin	0	4	1	14915	10955	117	\N	1	dps	\N
408	BG	BG4	Alliance	Warrior	2	1	22	21614	5073	551	1	\N	dps	\N
409	BG	BG4	Alliance	Shaman	0	0	39	7135	90943	612	1	\N	heal	\N
410	BG	BG4	Horde	Demon Hunter	0	2	1	5971	0	117	\N	1	dps	\N
411	BG	BG4	Alliance	Paladin	3	1	20	21950	6446	547	1	\N	dps	\N
412	BG	BG4	Alliance	Paladin	0	0	8	3575	3463	507	1	\N	dps	\N
413	BG	BG4	Horde	Warlock	0	5	1	27489	19149	117	\N	1	dps	\N
414	BG	BG4	Alliance	Warlock	11	0	38	75321	22441	608	1	\N	dps	\N
415	BG	BG5	Horde	Druid	0	0	20	1838	52542	358	1	\N	heal	\N
416	BG	BG5	Horde	Warrior	2	0	9	19349	315	487	1	\N	dps	\N
417	BG	BG5	Horde	Hunter	6	1	18	93288	7983	355	1	\N	dps	\N
418	BG	BG5	Alliance	Priest	0	4	1	13500	42184	205	\N	1	heal	\N
419	BG	BG5	Horde	Priest	2	0	22	28188	4095	514	1	\N	dps	\N
420	BG	BG5	Horde	Shaman	0	0	22	3021	62329	363	1	\N	heal	\N
421	BG	BG5	Horde	Demon Hunter	6	0	22	47380	7762	515	1	\N	dps	\N
422	BG	BG5	Alliance	Priest	2	1	3	47257	26763	211	\N	1	dps	\N
423	BG	BG5	Alliance	Shaman	0	1	3	25257	25648	211	\N	1	heal	\N
424	BG	BG5	Alliance	Mage	0	5	0	11521	10533	202	\N	1	dps	\N
425	BG	BG5	Horde	Shaman	0	0	14	3011	14531	498	1	\N	heal	\N
426	BG	BG5	Horde	Rogue	3	1	20	32965	2700	359	1	\N	dps	\N
427	BG	BG5	Alliance	Hunter	0	2	1	13386	2302	175	\N	1	dps	\N
428	BG	BG5	Alliance	Shaman	0	2	1	10012	19590	205	\N	1	heal	\N
429	BG	BG5	Alliance	Warrior	0	1	3	4461	105	211	\N	1	dps	\N
430	BG	BG5	Alliance	Paladin	0	1	3	6852	1536	211	\N	1	dps	\N
431	BG	BG5	Alliance	Hunter	0	2	2	51929	9739	208	\N	1	dps	\N
432	BG	BG5	Alliance	Hunter	1	3	3	33120	3651	211	\N	1	dps	\N
433	BG	BG5	Horde	Warlock	3	1	18	24069	11209	505	1	\N	dps	\N
434	BG	BG5	Horde	Druid	0	0	20	743	64975	359	1	\N	heal	\N
435	BG	BG6	Horde	Warrior	9	0	47	61047	8097	558	1	\N	dps	\N
436	BG	BG6	Alliance	Demon Hunter	0	8	0	52850	5448	187	\N	1	dps	\N
437	BG	BG6	Horde	Demon Hunter	3	0	37	51960	2985	388	1	\N	dps	\N
438	BG	BG6	Horde	Warrior	0	0	48	89098	12649	560	1	\N	dps	\N
439	BG	BG6	Horde	Shaman	0	0	32	2971	39918	528	1	\N	heal	\N
440	BG	BG6	Alliance	Demon Hunter	0	6	0	44644	6760	172	\N	1	dps	\N
441	BG	BG6	Horde	Death Knight	9	0	46	71646	16102	556	1	\N	dps	\N
442	BG	BG6	Alliance	Druid	0	1	0	0	32218	142	\N	1	heal	\N
443	BG	BG6	Horde	Mage	2	0	30	33017	5245	374	1	\N	dps	\N
444	BG	BG6	Alliance	Monk	0	4	0	1110	24319	187	\N	1	heal	\N
445	BG	BG6	Alliance	Warlock	0	9	0	71967	24561	187	\N	1	dps	\N
446	BG	BG6	Horde	Demon Hunter	1	0	17	10280	915	498	1	\N	dps	\N
447	BG	BG6	Alliance	Death Knight	0	2	0	14892	14134	142	\N	1	dps	\N
448	BG	BG6	Horde	Mage	4	0	48	61508	4299	560	1	\N	dps	\N
449	BG	BG6	Alliance	Demon Hunter	0	0	0	872	445	187	\N	1	dps	\N
450	BG	BG6	Horde	Paladin	11	0	47	54087	15989	558	1	\N	dps	\N
451	BG	BG6	Horde	Monk	0	0	47	10568	127000	558	1	\N	heal	\N
452	BG	BG6	Alliance	Priest	0	5	0	4206	37555	187	\N	1	heal	\N
453	BG	BG6	Alliance	Warrior	0	1	0	1818	0	127	\N	1	dps	\N
454	BG	BG6	Alliance	Death Knight	0	2	0	5683	5309	187	\N	1	dps	\N
455	BG	BG7	Alliance	Warrior	3	4	21	41100	23172	778	1	\N	dps	\N
456	BG	BG7	Horde	Druid	0	5	12	1118	99044	145	\N	1	heal	\N
457	BG	BG7	Horde	Hunter	4	4	16	33017	6854	150	\N	1	dps	\N
458	BG	BG7	Horde	Hunter	4	1	26	22624	7012	174	\N	1	dps	\N
459	BG	BG7	Alliance	Warlock	8	3	30	38779	17181	817	1	\N	dps	\N
460	BG	BG7	Horde	Hunter	5	3	22	41010	7769	161	\N	1	dps	\N
461	BG	BG7	Alliance	Rogue	3	2	19	24494	7590	561	1	\N	dps	\N
462	BG	BG7	Horde	Shaman	1	5	10	5254	50267	138	\N	1	heal	\N
463	BG	BG7	Alliance	Shaman	1	3	27	30826	9849	584	1	\N	dps	\N
464	BG	BG7	Horde	Mage	3	4	21	22158	12842	159	\N	1	dps	\N
465	BG	BG7	Horde	Mage	0	7	13	36633	6862	143	\N	1	dps	\N
466	BG	BG7	Alliance	Mage	2	4	9	38542	4321	743	1	\N	dps	\N
467	BG	BG7	Alliance	Paladin	13	1	34	86224	25128	832	1	\N	dps	\N
468	BG	BG7	Horde	Hunter	1	5	10	54158	9919	135	\N	1	dps	\N
469	BG	BG7	Horde	Druid	1	2	13	13693	20245	144	\N	1	heal	\N
470	BG	BG7	Horde	Demon Hunter	7	5	22	111000	15850	167	\N	1	dps	\N
471	BG	BG7	Alliance	Paladin	5	1	29	59524	26524	589	1	\N	dps	\N
472	BG	BG7	Alliance	Shaman	1	2	29	7425	51759	583	1	\N	heal	\N
473	BG	BG7	Alliance	Paladin	1	4	7	46376	15261	733	1	\N	dps	\N
474	BG	BG7	Alliance	Death Knight	2	3	25	61206	22649	790	1	\N	dps	\N
475	BG	BG8	Horde	Warlock	3	0	22	48823	14259	670	1	\N	dps	1
476	BG	BG8	Horde	Warrior	3	3	17	48932	15757	667	1	\N	dps	1
477	BG	BG8	Horde	Mage	4	1	29	24388	2596	683	1	\N	dps	1
478	BG	BG8	Alliance	Priest	1	4	9	17734	8223	288	\N	1	dps	1
479	BG	BG8	Horde	Mage	6	1	24	67905	7930	670	1	\N	dps	1
480	BG	BG8	Horde	Warrior	6	1	21	53403	16902	674	1	\N	dps	1
481	BG	BG8	Alliance	Priest	3	5	7	36252	16261	281	\N	1	dps	1
482	BG	BG8	Horde	Shaman	0	2	27	10925	68694	680	1	\N	heal	1
483	BG	BG8	Horde	Hunter	5	0	15	21996	0	656	1	\N	dps	1
484	BG	BG8	Alliance	Paladin	0	2	5	5778	5784	258	\N	1	heal	1
485	BG	BG8	Alliance	Druid	0	1	11	1256	84714	297	\N	1	heal	1
486	BG	BG8	Alliance	Paladin	0	5	6	51890	12249	278	\N	1	dps	1
487	BG	BG8	Horde	Druid	4	2	23	55751	13024	669	1	\N	dps	1
488	BG	BG8	Horde	Paladin	0	1	27	6631	55709	679	1	\N	heal	1
489	BG	BG8	Alliance	Warrior	2	4	7	27514	4503	281	\N	1	dps	1
490	BG	BG8	Alliance	Monk	2	4	9	40972	26314	290	\N	1	dps	1
491	BG	BG8	Alliance	Mage	2	2	11	43490	11718	297	\N	1	dps	1
492	BG	BG8	Horde	Shaman	4	2	25	61060	1578	673	1	\N	dps	1
493	BG	BG8	Alliance	Hunter	0	1	5	12015	4128	242	\N	1	dps	1
494	BG	BG9	Alliance	Hunter	5	5	20	73062	12905	989	1	\N	dps	1
495	BG	BG9	Horde	Death Knight	4	7	30	95493	29129	296	\N	1	dps	1
496	BG	BG9	Alliance	Druid	0	4	45	224	215000	1077	1	\N	heal	1
497	BG	BG9	Horde	Druid	2	3	28	49531	13027	291	\N	1	dps	1
498	BG	BG9	Alliance	Warrior	4	3	26	63651	31561	1007	1	\N	dps	1
499	BG	BG9	Horde	Shaman	1	7	28	8938	99399	290	\N	1	heal	1
500	BG	BG9	Horde	Mage	7	5	28	127000	23979	293	\N	1	dps	1
501	BG	BG9	Horde	Warrior	3	5	25	44336	6910	296	\N	1	dps	1
502	BG	BG9	Alliance	Rogue	11	3	35	45332	8293	1037	1	\N	dps	1
503	BG	BG9	Alliance	Warrior	6	4	38	100000	12070	1048	1	\N	dps	1
504	BG	BG9	Alliance	Mage	8	3	39	96730	7997	1054	1	\N	dps	1
505	BG	BG9	Horde	Monk	0	5	28	3990	101000	289	\N	1	heal	1
506	BG	BG9	Alliance	Warlock	8	5	38	86617	52555	1054	1	\N	dps	1
507	BG	BG9	Horde	Paladin	9	4	35	192000	46517	315	\N	1	dps	1
508	BG	BG9	Horde	Death Knight	5	5	25	70827	20915	289	\N	1	dps	1
509	BG	BG9	Horde	Warrior	2	7	21	52432	4377	279	\N	1	dps	1
510	BG	BG9	Alliance	Rogue	4	2	35	58944	4813	1035	1	\N	dps	1
511	BG	BG9	Horde	Death Knight	4	4	16	39249	26497	276	\N	1	dps	1
512	BG	BG9	Alliance	Shaman	1	4	35	44200	142000	698	1	\N	heal	1
513	BG	BG9	Alliance	Warlock	3	7	25	82345	29406	672	1	\N	dps	1
514	BG	BG10	Horde	Hunter	4	0	8	28773	3928	336	1	\N	dps	\N
515	BG	BG10	Horde	Druid	0	0	22	1818	179000	514	1	\N	heal	\N
516	BG	BG10	Horde	Demon Hunter	4	0	21	66986	7942	514	1	\N	dps	\N
517	BG	BG10	Alliance	Hunter	0	2	0	16669	4304	172	\N	1	dps	\N
518	BG	BG10	Alliance	Hunter	0	4	7	50752	13190	190	\N	1	dps	\N
519	BG	BG10	Horde	Shaman	0	2	19	10968	88406	512	1	\N	heal	\N
520	BG	BG10	Alliance	Warrior	1	0	2	12938	2906	179	\N	1	dps	\N
521	BG	BG10	Alliance	Warrior	1	3	3	8580	3849	183	\N	1	dps	\N
522	BG	BG10	Alliance	Druid	0	1	3	61503	755	182	\N	1	dps	\N
523	BG	BG10	Horde	Warlock	3	2	21	57596	20110	514	1	\N	dps	\N
524	BG	BG10	Horde	Warlock	3	0	20	71472	17020	516	1	\N	dps	\N
525	BG	BG10	Alliance	Paladin	0	2	1	0	56983	160	\N	1	heal	\N
526	BG	BG10	Alliance	Warlock	3	3	9	96378	29466	197	\N	1	dps	\N
527	BG	BG10	Horde	Warrior	5	2	14	49937	14438	378	1	\N	dps	\N
528	BG	BG10	Horde	Paladin	4	1	21	69992	25463	362	1	\N	dps	\N
529	BG	BG10	Alliance	Priest	1	2	5	27113	77543	185	\N	1	heal	\N
530	BG	BG10	Alliance	Druid	0	4	3	3724	6158	183	\N	1	heal	\N
531	BG	BG10	Horde	Demon Hunter	4	2	21	40051	6659	363	1	\N	dps	\N
532	BG	BG10	Alliance	Warrior	1	2	3	17515	410	182	\N	1	dps	\N
533	BG	BG10	Horde	Demon Hunter	1	1	21	26795	1614	512	1	\N	dps	\N
534	BG	BG11	Horde	Death Knight	1	0	14	17038	11942	347	1	\N	dps	\N
535	BG	BG11	Horde	Hunter	0	0	14	31409	2725	497	1	\N	dps	\N
536	BG	BG11	Horde	Paladin	2	0	21	27294	15874	511	1	\N	dps	\N
537	BG	BG11	Horde	Warrior	6	1	16	23478	2245	501	1	\N	dps	\N
538	BG	BG11	Horde	Warrior	6	1	18	26419	5497	497	1	\N	dps	\N
539	BG	BG11	Horde	Paladin	2	0	21	1838	52741	511	1	\N	heal	\N
540	BG	BG11	Alliance	Paladin	0	2	3	55422	11236	166	\N	1	dps	\N
541	BG	BG11	Horde	Shaman	0	0	15	8015	64070	351	1	\N	heal	\N
542	BG	BG11	Horde	Warrior	1	2	14	23118	2056	349	1	\N	dps	\N
543	BG	BG11	Horde	Demon Hunter	1	0	21	21644	10471	511	1	\N	dps	\N
544	BG	BG11	Horde	Warrior	1	0	9	9409	1033	489	1	\N	dps	\N
545	BG	BG11	Alliance	Hunter	1	1	3	8058	2416	167	\N	1	dps	\N
546	BG	BG11	Alliance	Rogue	0	1	2	4373	1069	166	\N	1	dps	\N
547	BG	BG11	Alliance	Hunter	1	1	1	6121	1327	161	\N	1	dps	\N
548	BG	BG11	Alliance	Monk	1	5	3	31448	14585	167	\N	1	dps	\N
549	BG	BG11	Alliance	Hunter	0	6	4	31924	5502	171	\N	1	dps	\N
550	BG	BG12	Horde	Warrior	3	4	8	38457	5352	146	\N	1	dps	\N
551	BG	BG12	Alliance	Druid	3	4	28	34703	6249	557	1	\N	dps	\N
552	BG	BG12	Horde	Warrior	1	5	9	31721	1171	148	\N	1	dps	\N
553	BG	BG12	Horde	Rogue	1	4	8	29680	10595	146	\N	1	dps	\N
554	BG	BG12	Horde	Shaman	0	7	8	14569	64481	146	\N	1	heal	\N
555	BG	BG12	Horde	Rogue	1	4	9	27807	1661	148	\N	1	dps	\N
556	BG	BG12	Horde	Priest	1	5	9	24202	63842	148	\N	1	heal	\N
557	BG	BG12	Horde	Priest	0	3	8	13578	6885	146	\N	1	dps	\N
558	BG	BG12	Alliance	Priest	6	2	25	85170	22911	775	1	\N	dps	\N
559	BG	BG12	Horde	Druid	1	0	7	23191	10470	144	\N	1	dps	\N
560	BG	BG12	Alliance	Warrior	1	0	16	10206	1572	758	1	\N	dps	\N
561	BG	BG12	Alliance	Warrior	12	0	36	41656	3957	811	1	\N	dps	\N
562	BG	BG12	Horde	Rogue	1	4	8	30254	9574	146	\N	1	dps	\N
563	BG	BG12	Horde	Hunter	0	1	7	21819	1979	144	\N	1	dps	\N
564	BG	BG12	Alliance	Paladin	0	0	21	7749	52641	762	1	\N	heal	\N
565	BG	BG12	Alliance	Mage	4	0	20	39708	409	760	1	\N	dps	\N
566	BG	BG12	Alliance	Warrior	3	1	25	13093	174	777	1	\N	dps	\N
567	BG	BG12	Alliance	Warlock	4	0	36	75698	14705	811	1	\N	dps	\N
568	BG	BG12	Alliance	Warrior	4	1	36	63787	7516	811	1	\N	dps	\N
569	BG	BG12	Alliance	Monk	0	1	32	516	117000	795	1	\N	heal	\N
570	BG	BG13	Horde	Warrior	2	4	29	19886	2915	532	1	\N	dps	\N
571	BG	BG13	Alliance	Paladin	0	3	16	16158	8072	201	\N	1	dps	\N
572	BG	BG13	Horde	Demon Hunter	9	3	36	17010	13374	400	1	\N	dps	\N
573	BG	BG13	Alliance	Rogue	2	6	17	10223	1022	204	\N	1	dps	\N
574	BG	BG13	Alliance	Demon Hunter	2	5	27	18184	1077	224	\N	1	dps	\N
575	BG	BG13	Alliance	Monk	9	5	31	26001	18171	236	\N	1	dps	\N
576	BG	BG13	Horde	Warlock	3	3	32	27319	20566	388	1	\N	dps	\N
577	BG	BG13	Horde	Warrior	2	3	7	7190	72	486	1	\N	dps	\N
578	BG	BG13	Horde	Shaman	2	2	21	18661	3799	503	1	\N	dps	\N
579	BG	BG13	Horde	Hunter	8	4	33	47819	2398	391	1	\N	dps	\N
580	BG	BG13	Alliance	Hunter	7	2	28	32654	1827	230	\N	1	dps	\N
581	BG	BG13	Alliance	Death Knight	3	3	26	26450	8498	226	\N	1	dps	\N
582	BG	BG13	Horde	Shaman	1	6	27	5136	47706	525	1	\N	heal	\N
583	BG	BG13	Alliance	Paladin	1	3	14	3918	5727	197	\N	1	heal	\N
584	BG	BG13	Alliance	Druid	2	3	20	28567	6885	212	\N	1	dps	\N
585	BG	BG13	Horde	Shaman	4	5	29	17411	6764	529	1	\N	dps	\N
586	BG	BG13	Horde	Priest	5	4	24	25677	14269	369	1	\N	dps	\N
587	BG	BG13	Alliance	Paladin	1	3	27	3320	56390	224	\N	1	heal	\N
588	BG	BG13	Horde	Paladin	2	0	12	8004	1824	505	1	\N	dps	\N
589	BG	BG13	Alliance	Demon Hunter	7	5	29	43702	2318	232	\N	1	dps	\N
590	BG	BG14	Alliance	Mage	2	3	16	17842	4086	509	1	\N	dps	\N
591	BG	BG14	Horde	Warrior	2	5	16	12875	9739	173	\N	1	dps	\N
592	BG	BG14	Alliance	Hunter	10	2	37	39952	6818	551	1	\N	dps	\N
593	BG	BG14	Alliance	Warlock	2	3	28	19670	15925	382	1	\N	dps	\N
594	BG	BG14	Alliance	Demon Hunter	1	4	27	20314	2488	374	1	\N	dps	\N
595	BG	BG14	Horde	Warrior	5	3	16	12360	897	177	\N	1	dps	\N
596	BG	BG14	Alliance	Druid	0	3	20	1449	34559	361	1	\N	heal	\N
597	BG	BG14	Horde	Demon Hunter	6	3	22	27615	5366	193	\N	1	dps	\N
598	BG	BG14	Alliance	Warrior	8	3	36	33178	2201	394	1	\N	dps	\N
599	BG	BG14	Horde	Shaman	1	6	22	7179	42196	194	\N	1	heal	\N
600	BG	BG14	Horde	Warlock	2	5	21	22808	13292	189	\N	1	dps	\N
601	BG	BG14	Alliance	Druid	10	1	44	43183	8469	565	1	\N	dps	\N
602	BG	BG14	Alliance	Warrior	8	1	32	17498	3366	538	1	\N	dps	\N
603	BG	BG14	Alliance	Paladin	5	2	40	33816	12701	560	1	\N	dps	\N
604	BG	BG14	Horde	Death Knight	4	7	18	29558	13978	183	\N	1	dps	\N
605	BG	BG14	Alliance	Death Knight	1	7	15	15382	8608	504	1	\N	dps	\N
606	BG	BG14	Horde	Druid	2	4	22	29908	3776	189	\N	1	dps	\N
607	BG	BG14	Horde	Death Knight	4	4	18	22348	12307	180	\N	1	dps	\N
608	BG	BG14	Horde	Rogue	1	5	13	11851	2518	165	\N	1	dps	\N
609	BG	BG14	Horde	Paladin	1	5	17	2312	24190	177	\N	1	heal	\N
610	BG	BG15	Horde	Rogue	7	0	25	13547	2293	378	1	\N	dps	\N
611	BG	BG15	Horde	Hunter	5	1	28	16167	1690	379	1	\N	dps	\N
612	BG	BG15	Horde	Paladin	1	1	8	10903	4449	485	1	\N	dps	\N
613	BG	BG15	Alliance	Priest	0	2	2	1130	13428	139	\N	1	heal	\N
614	BG	BG15	Alliance	Rogue	0	3	1	964	2071	139	\N	1	dps	\N
615	BG	BG15	Horde	Shaman	0	0	28	3740	36886	527	1	\N	heal	\N
616	BG	BG15	Horde	Rogue	1	0	20	13687	1538	361	1	\N	dps	\N
617	BG	BG15	Alliance	Hunter	1	1	2	2666	402	110	\N	1	dps	\N
618	BG	BG15	Alliance	Rogue	0	2	4	3686	3172	144	\N	1	dps	\N
619	BG	BG15	Horde	Paladin	7	0	26	21372	7173	371	1	\N	dps	\N
620	BG	BG15	Alliance	Mage	1	5	3	13910	8891	141	\N	1	dps	\N
621	BG	BG15	Alliance	Rogue	1	6	3	16141	2741	144	\N	1	dps	\N
622	BG	BG15	Alliance	Warlock	0	4	3	23924	17004	141	\N	1	dps	\N
623	BG	BG15	Horde	Rogue	0	2	17	8964	854	505	1	\N	dps	\N
624	BG	BG15	Horde	Warlock	4	0	27	21642	9028	523	1	\N	dps	\N
625	BG	BG15	Alliance	Shaman	1	5	3	7476	2542	134	\N	1	dps	\N
626	BG	BG15	Alliance	Priest	1	2	2	9393	8427	139	\N	1	dps	\N
627	BG	BG15	Horde	Druid	3	0	24	14931	3200	517	1	\N	dps	\N
628	BG	BG15	Alliance	Druid	0	1	1	3326	801	137	\N	1	dps	\N
629	BG	BG15	Horde	Paladin	4	1	23	17228	7512	374	1	\N	dps	\N
630	BG	BG16	Alliance	Druid	0	4	9	524	23763	145	\N	1	heal	\N
631	BG	BG16	Horde	Death Knight	5	1	27	25186	14420	523	1	\N	dps	\N
632	BG	BG16	Horde	Demon Hunter	1	4	22	18539	594	364	1	\N	dps	\N
633	BG	BG16	Alliance	Demon Hunter	1	3	9	13985	2410	144	\N	1	dps	\N
634	BG	BG16	Alliance	Mage	0	5	9	14465	6638	144	\N	1	dps	\N
635	BG	BG16	Horde	Shaman	0	2	27	3469	31656	523	1	\N	heal	\N
636	BG	BG16	Alliance	Death Knight	1	3	10	11823	5543	147	\N	1	dps	\N
637	BG	BG16	Alliance	Druid	0	2	5	263	19462	135	\N	1	heal	\N
638	BG	BG16	Horde	Paladin	6	2	11	21348	7193	365	1	\N	dps	\N
639	BG	BG16	Alliance	Druid	1	1	8	16602	3269	142	\N	1	dps	\N
640	BG	BG16	Horde	Demon Hunter	1	0	12	15571	3013	495	1	\N	dps	\N
641	BG	BG16	Horde	Mage	3	0	25	8107	3011	370	1	\N	dps	\N
642	BG	BG16	Alliance	Rogue	1	2	5	6613	1813	135	\N	1	dps	\N
643	BG	BG16	Horde	Demon Hunter	5	2	26	23585	5757	371	1	\N	dps	\N
644	BG	BG16	Alliance	Paladin	3	3	12	17675	5534	152	\N	1	dps	\N
645	BG	BG16	Horde	Rogue	2	1	16	7612	2420	503	1	\N	dps	\N
646	BG	BG16	Alliance	Hunter	2	5	8	16273	3274	143	\N	1	dps	\N
647	BG	BG16	Horde	Demon Hunter	8	0	32	28829	4829	385	1	\N	dps	\N
648	BG	BG16	Alliance	Warrior	2	5	5	16207	5416	137	\N	1	dps	\N
649	BG	BG16	Horde	Priest	1	1	29	8091	13856	379	1	\N	heal	\N
650	TK	TK1	Alliance	Death Knight	6	3	58	25925	11314	554	1	\N	dps	\N
651	TK	TK1	Alliance	Mage	6	2	57	28438	5258	251	1	\N	dps	\N
652	TK	TK1	Alliance	Druid	8	3	55	45584	7932	547	1	\N	dps	\N
653	TK	TK1	Alliance	Shaman	0	0	59	7664	54796	405	1	\N	heal	\N
654	TK	TK1	Horde	Warrior	2	6	12	32693	7929	185	\N	1	dps	\N
655	TK	TK1	Alliance	Paladin	8	0	63	36252	11032	413	1	\N	dps	\N
656	TK	TK1	Alliance	Druid	0	0	63	519	112000	563	1	\N	heal	\N
657	TK	TK1	Horde	Warlock	0	7	11	26600	16275	183	\N	1	dps	\N
658	TK	TK1	Horde	Mage	0	8	11	29222	19680	183	\N	1	dps	\N
659	TK	TK1	Horde	Paladin	1	8	13	30888	4536	187	\N	1	dps	\N
660	TK	TK1	Horde	Demon Hunter	3	6	12	53560	3941	185	\N	1	dps	\N
661	TK	TK1	Horde	Shaman	1	7	13	4277	26203	187	\N	1	heal	\N
662	TK	TK1	Horde	Hunter	3	4	14	55258	1439	189	\N	1	dps	\N
663	TK	TK1	Alliance	Warlock	7	2	53	30572	25501	396	1	\N	dps	\N
664	TK	TK1	Alliance	Warrior	12	1	59	50490	11310	555	1	\N	dps	\N
665	TK	TK1	Horde	Demon Hunter	4	7	11	44185	4955	183	\N	1	dps	\N
666	TK	TK1	Alliance	Paladin	3	3	62	21305	5853	561	1	\N	dps	\N
667	TK	TK1	Horde	Monk	0	4	11	182	19596	183	\N	1	heal	\N
668	TK	TK1	Horde	Warlock	0	6	12	7665	7230	185	\N	1	dps	\N
669	TK	TK1	Alliance	Rogue	12	0	63	40777	2773	413	1	\N	dps	\N
670	TK	TK2	Alliance	Warrior	7	3	39	47600	15357	562	1	\N	dps	\N
671	TK	TK2	Alliance	Druid	0	0	40	727	98081	789	1	\N	heal	\N
672	TK	TK2	Horde	Monk	0	4	11	4183	77442	157	\N	1	heal	\N
673	TK	TK2	Horde	Death Knight	1	3	14	36888	7085	163	\N	1	dps	\N
674	TK	TK2	Horde	Druid	1	7	9	43025	5737	153	\N	1	dps	\N
675	TK	TK2	Alliance	Paladin	7	0	40	79804	11908	564	1	\N	dps	\N
676	TK	TK2	Horde	Shaman	0	3	9	6092	28164	153	\N	1	heal	\N
677	TK	TK2	Alliance	Warrior	9	2	38	58892	13942	784	1	\N	dps	\N
678	TK	TK2	Alliance	Hunter	1	3	35	30043	1733	551	1	\N	dps	\N
679	TK	TK2	Horde	Warlock	1	1	10	54384	15453	145	\N	1	dps	\N
680	TK	TK2	Horde	Warrior	4	6	13	51673	3985	161	\N	1	dps	\N
681	TK	TK2	Horde	Warrior	1	4	14	43972	21916	163	\N	1	dps	\N
682	TK	TK2	Horde	Warrior	3	5	10	52674	5553	155	\N	1	dps	\N
683	TK	TK2	Alliance	Druid	1	0	40	2881	76860	564	1	\N	heal	\N
684	TK	TK2	Alliance	Mage	4	0	40	39389	2558	564	1	\N	dps	\N
685	TK	TK2	Horde	Warrior	1	6	10	40636	7398	155	\N	1	dps	\N
686	TK	TK2	Alliance	Death Knight	5	2	39	46325	38957	787	1	\N	dps	\N
687	TK	TK2	Horde	Shaman	0	0	6	420	21512	147	\N	1	heal	\N
688	TK	TK2	Alliance	Death Knight	2	3	37	52797	10996	556	1	\N	dps	\N
689	TK	TK2	Alliance	Druid	4	2	35	55274	10055	551	1	\N	dps	\N
690	TK	TK3	Alliance	Paladin	1	3	19	28403	9156	260	\N	1	dps	\N
691	TK	TK3	Horde	Demon Hunter	4	4	36	33317	8092	532	1	\N	dps	\N
692	TK	TK3	Horde	Demon Hunter	3	2	41	15656	3075	542	1	\N	dps	\N
693	TK	TK3	Horde	Mage	4	1	42	32779	8813	544	1	\N	dps	\N
694	TK	TK3	Horde	Death Knight	5	2	38	27892	21767	536	1	\N	dps	\N
695	TK	TK3	Alliance	Shaman	2	7	16	31373	2979	250	\N	1	dps	\N
696	TK	TK3	Horde	Warlock	6	4	35	34928	21698	380	1	\N	dps	\N
697	TK	TK3	Horde	Hunter	6	2	40	36875	3791	540	1	\N	dps	\N
698	TK	TK3	Alliance	Monk	4	6	20	40171	15493	262	\N	1	dps	\N
699	TK	TK3	Horde	Shaman	0	3	36	9444	115000	532	1	\N	heal	\N
700	TK	TK3	Alliance	Mage	3	5	14	80459	25562	243	\N	1	dps	\N
701	TK	TK3	Alliance	Warrior	6	6	18	58965	10083	261	\N	1	dps	\N
702	TK	TK3	Alliance	Demon Hunter	2	3	21	18810	10863	267	\N	1	dps	\N
703	TK	TK3	Alliance	Warrior	1	2	7	27430	3198	163	\N	1	dps	\N
704	TK	TK3	Alliance	Warlock	1	2	4	14054	3579	154	\N	1	dps	\N
705	TK	TK3	Horde	Demon Hunter	5	2	38	42652	6062	536	1	\N	dps	\N
706	TK	TK3	Alliance	Paladin	2	4	17	17365	6644	256	\N	1	dps	\N
707	TK	TK3	Alliance	Rogue	1	5	18	40260	1691	258	\N	1	dps	\N
708	TK	TK3	Horde	Warlock	3	3	42	38129	18149	544	1	\N	dps	\N
709	TK	TK3	Horde	Paladin	6	0	44	56388	19717	548	1	\N	dps	\N
710	TK	TK4	Alliance	Paladin	15	3	51	110000	25344	406	\N	1	dps	\N
711	TK	TK4	Alliance	Warlock	12	2	53	107000	33890	414	\N	1	dps	\N
712	TK	TK4	Horde	Shaman	10	4	41	89306	13354	535	1	\N	dps	\N
713	TK	TK4	Alliance	Hunter	9	5	48	111000	8083	399	\N	1	dps	\N
714	TK	TK4	Horde	Paladin	8	6	39	71519	14853	531	1	\N	dps	\N
715	TK	TK4	Horde	Warrior	8	5	41	57200	10280	385	1	\N	dps	\N
716	TK	TK4	Horde	Druid	7	7	38	56858	4410	529	1	\N	dps	\N
717	TK	TK4	Alliance	Death Knight	6	3	51	62616	20198	406	\N	1	dps	\N
718	TK	TK4	Alliance	Death Knight	5	7	45	52500	14125	391	\N	1	dps	\N
719	TK	TK4	Horde	Mage	4	4	41	75821	18728	535	1	\N	dps	\N
720	TK	TK4	Alliance	Warrior	3	7	40	49141	5890	378	\N	1	dps	\N
721	TK	TK4	Horde	Rogue	3	5	38	58952	3635	529	1	\N	dps	\N
722	TK	TK4	Horde	Druid	2	6	34	24641	9937	521	1	\N	dps	\N
723	TK	TK4	Alliance	Mage	2	6	42	40390	12718	388	\N	1	dps	\N
724	TK	TK4	Alliance	Priest	1	1	54	32768	95948	417	\N	1	heal	\N
725	TK	TK4	Horde	Shaman	0	8	35	10532	109000	523	1	\N	heal	\N
726	TK	TK4	Alliance	Paladin	0	5	44	8162	31777	391	\N	1	heal	\N
727	TK	TK4	Horde	Death Knight	0	5	35	30200	50294	373	1	\N	dps	\N
728	TK	TK4	Alliance	Mage	0	4	41	33043	5391	383	\N	1	dps	\N
729	TK	TK4	Horde	Druid	0	4	38	842	98411	379	1	\N	heal	\N
730	TK	TK5	Alliance	Hunter	5	2	27	52828	4933	337	\N	1	dps	\N
731	TK	TK5	Horde	Demon Hunter	8	1	56	78372	9397	563	1	\N	dps	\N
732	TK	TK5	Alliance	Demon Hunter	2	8	17	67018	13042	309	\N	1	dps	\N
733	TK	TK5	Horde	Warrior	11	3	50	61415	14373	401	1	\N	dps	\N
734	TK	TK5	Horde	Demon Hunter	7	5	48	44773	6194	395	1	\N	dps	\N
735	TK	TK5	Alliance	Death Knight	4	7	19	47193	17033	308	\N	1	dps	\N
736	TK	TK5	Alliance	Druid	4	3	26	85965	12005	336	\N	1	dps	\N
737	TK	TK5	Horde	Warlock	4	5	49	53797	32985	393	1	\N	dps	\N
738	TK	TK5	Horde	Shaman	0	4	52	19266	76639	555	1	\N	heal	\N
739	TK	TK5	Alliance	Mage	0	7	21	16805	5233	318	\N	1	dps	\N
740	TK	TK5	Alliance	Priest	0	6	19	2014	20466	309	\N	1	heal	\N
741	TK	TK5	Alliance	Shaman	1	3	26	50143	13530	334	\N	1	dps	\N
742	TK	TK5	Horde	Druid	1	5	41	23257	10814	531	1	\N	dps	\N
743	TK	TK5	Horde	Priest	3	0	56	20153	152000	413	1	\N	heal	\N
744	TK	TK5	Horde	Rogue	5	3	52	57792	2635	551	1	\N	dps	\N
745	TK	TK5	Horde	Druid	3	3	49	16615	12417	549	1	\N	dps	\N
746	TK	TK5	Alliance	Priest	2	5	22	14164	51093	323	\N	1	heal	\N
747	TK	TK5	Alliance	Warrior	5	7	20	66317	12336	317	\N	1	dps	\N
748	TK	TK5	Alliance	Warrior	2	7	22	56111	6297	318	\N	1	dps	\N
749	TK	TK5	Horde	Warrior	15	0	56	88713	14976	413	1	\N	dps	\N
750	TK	TK6	Alliance	Demon Hunter	4	6	19	40144	2364	165	\N	1	dps	\N
751	TK	TK6	Horde	Mage	0	2	37	19317	4111	520	1	\N	dps	\N
752	TK	TK6	Alliance	Paladin	1	5	20	1450	21173	169	\N	1	heal	\N
753	TK	TK6	Alliance	Warlock	3	3	25	24614	11955	179	\N	1	dps	\N
754	TK	TK6	Alliance	Mage	0	4	16	24478	5175	157	\N	1	dps	\N
755	TK	TK6	Horde	Mage	3	3	37	16050	5989	518	1	\N	dps	\N
756	TK	TK6	Horde	Warrior	3	2	39	18658	10170	522	1	\N	dps	\N
757	TK	TK6	Horde	Shaman	0	2	39	4783	49940	522	1	\N	heal	\N
758	TK	TK6	Horde	Rogue	7	2	35	21936	4953	366	1	\N	dps	\N
759	TK	TK6	Alliance	Demon Hunter	1	6	16	12232	4278	163	\N	1	dps	\N
760	TK	TK6	Alliance	Death Knight	2	4	23	23452	24221	173	\N	1	dps	\N
761	TK	TK6	Horde	Rogue	7	2	36	16219	4135	516	1	\N	dps	\N
762	TK	TK6	Horde	Mage	6	3	39	25811	5031	522	1	\N	dps	\N
763	TK	TK6	Horde	Warlock	7	2	39	25832	16403	374	1	\N	dps	\N
764	TK	TK6	Alliance	Hunter	5	4	23	15034	1119	173	\N	1	dps	\N
765	TK	TK6	Alliance	Druid	0	2	18	417	593	165	\N	1	heal	\N
766	TK	TK6	Alliance	Rogue	5	3	21	15941	3727	167	\N	1	dps	\N
767	TK	TK6	Horde	Warrior	5	5	37	16834	0	368	1	\N	dps	\N
768	TK	TK6	Horde	Rogue	2	2	41	18018	3654	528	1	\N	dps	\N
769	TK	TK6	Alliance	Warlock	2	4	23	8873	11163	173	\N	1	dps	\N
770	TK	TK7	Horde	Rogue	4	2	30	21243	2127	508	1	\N	dps	\N
771	TK	TK7	Alliance	Druid	7	4	28	29090	7893	193	\N	1	dps	\N
772	TK	TK7	Horde	Rogue	2	3	35	18594	1540	368	1	\N	dps	\N
773	TK	TK7	Horde	Warrior	10	3	34	23982	982	366	1	\N	dps	\N
774	TK	TK7	Alliance	Druid	0	2	29	657	29445	195	\N	1	heal	\N
775	TK	TK7	Alliance	Mage	2	5	25	18870	4399	187	\N	1	dps	\N
776	TK	TK7	Horde	Warrior	6	3	33	29098	1119	364	1	\N	dps	\N
777	TK	TK7	Horde	Shaman	0	4	33	7964	36776	513	1	\N	heal	\N
778	TK	TK7	Alliance	Shaman	0	7	24	459	21956	185	\N	1	heal	\N
779	TK	TK7	Alliance	Demon Hunter	8	4	28	39718	4707	193	\N	1	dps	\N
780	TK	TK7	Horde	Druid	2	2	34	11218	2420	365	1	\N	dps	\N
781	TK	TK7	Alliance	Rogue	4	1	30	15663	1121	197	\N	1	dps	\N
782	TK	TK7	Alliance	Druid	1	2	30	408	33430	197	\N	1	heal	\N
783	TK	TK7	Alliance	Shaman	0	0	30	7165	35460	197	\N	1	heal	\N
784	TK	TK7	Horde	Demon Hunter	5	4	34	54877	2465	366	1	\N	dps	\N
785	TK	TK7	Horde	Monk	3	3	30	33851	10463	507	1	\N	dps	\N
786	TK	TK7	Horde	Warrior	3	3	33	22460	2094	513	1	\N	dps	\N
787	TK	TK7	Horde	Monk	0	3	34	811	29869	515	1	\N	heal	\N
788	TK	TK7	Alliance	Warrior	7	5	29	18734	743	195	\N	1	dps	\N
789	TK	TK7	Alliance	Demon Hunter	0	4	29	14271	6498	195	\N	1	dps	\N
790	TK	TK8	Alliance	Paladin	0	0	46	776	44074	389	1	\N	heal	\N
791	TK	TK8	Horde	Demon Hunter	1	5	12	22140	3990	136	\N	1	dps	\N
792	TK	TK8	Alliance	Warlock	5	0	46	23575	9552	453	1	\N	dps	\N
793	TK	TK8	Alliance	Demon Hunter	5	1	43	14415	4208	383	1	\N	dps	\N
794	TK	TK8	Alliance	Demon Hunter	2	3	41	11999	954	379	1	\N	dps	\N
795	TK	TK8	Alliance	Rogue	7	2	41	15652	680	529	1	\N	dps	\N
796	TK	TK8	Horde	Shaman	0	6	11	2161	10910	134	\N	1	heal	\N
797	TK	TK8	Horde	Shaman	0	5	8	1813	789	128	\N	1	dps	\N
798	TK	TK8	Horde	Rogue	2	6	11	11151	1511	135	\N	1	dps	\N
799	TK	TK8	Alliance	Hunter	3	2	45	23767	16689	445	1	\N	dps	\N
800	TK	TK8	Alliance	Warrior	8	2	39	9592	2314	375	1	\N	dps	\N
801	TK	TK8	Horde	Mage	1	6	9	14868	684	130	\N	1	dps	\N
802	TK	TK8	Horde	Rogue	0	3	11	6714	858	135	\N	1	dps	\N
803	TK	TK8	Alliance	Rogue	3	2	39	11241	1791	435	1	\N	dps	\N
804	TK	TK8	Alliance	Paladin	7	0	46	23275	7033	389	1	\N	dps	\N
805	TK	TK8	Horde	Warlock	2	4	9	22391	9911	130	\N	1	dps	\N
806	TK	TK8	Horde	Rogue	5	5	13	20230	3127	139	\N	1	dps	\N
807	TK	TK8	Alliance	Warrior	6	1	43	21528	3270	442	1	\N	dps	\N
808	TK	TK8	Horde	Mage	2	4	13	11738	6312	139	\N	1	dps	\N
809	TK	TK9	Horde	Monk	10	7	26	37017	1028	232	\N	1	dps	\N
810	TK	TK9	Alliance	Mage	1	2	51	23153	4868	409	1	\N	dps	\N
811	TK	TK9	Horde	Rogue	4	6	25	14648	1417	230	\N	1	dps	\N
812	TK	TK9	Alliance	Paladin	2	5	40	14447	4293	400	1	\N	dps	\N
813	TK	TK9	Alliance	Warrior	4	3	51	20146	954	409	1	\N	dps	\N
814	TK	TK9	Horde	Hunter	2	5	28	24702	5084	236	\N	1	dps	\N
815	TK	TK9	Horde	Warrior	3	7	20	14067	1804	220	\N	1	dps	\N
816	TK	TK9	Alliance	Mage	7	1	59	26239	3853	425	1	\N	dps	\N
817	TK	TK9	Horde	Demon Hunter	2	5	26	32412	5569	232	\N	1	dps	\N
818	TK	TK9	Alliance	Death Knight	7	6	51	27515	13645	559	1	\N	dps	\N
819	TK	TK9	Horde	Shaman	1	2	12	6719	1310	138	\N	1	dps	\N
820	TK	TK9	Horde	Shaman	1	6	27	4967	31241	234	\N	1	heal	\N
821	TK	TK9	Alliance	Shaman	12	3	52	31495	11740	411	1	\N	dps	\N
822	TK	TK9	Alliance	Rogue	6	3	54	27217	1183	565	1	\N	dps	\N
823	TK	TK9	Alliance	Rogue	11	2	57	17946	2051	421	1	\N	dps	\N
824	TK	TK9	Alliance	Paladin	10	3	55	35660	9172	417	1	\N	dps	\N
825	TK	TK9	Horde	Priest	5	7	28	33481	11695	236	\N	1	dps	\N
826	TK	TK9	Horde	Paladin	0	4	27	8769	5616	234	\N	1	dps	\N
827	TK	TK9	Horde	Rogue	4	8	27	20585	70	234	\N	1	dps	\N
828	TK	TK9	Alliance	Monk	0	4	55	105	66324	567	1	\N	heal	\N
829	TK	TK10	Alliance	Priest	4	6	28	45014	8505	226	\N	1	dps	\N
830	TK	TK10	Alliance	Hunter	2	7	28	19295	4850	229	\N	1	dps	\N
831	TK	TK10	Horde	Monk	5	4	45	13002	8463	528	1	\N	dps	\N
832	TK	TK10	Alliance	Rogue	2	3	31	17239	2686	230	\N	1	dps	\N
833	TK	TK10	Alliance	Warlock	2	4	11	15464	5659	147	\N	1	dps	\N
834	TK	TK10	Horde	Death Knight	3	2	49	10892	10115	546	1	\N	dps	\N
835	TK	TK10	Horde	Paladin	8	1	50	41327	12071	398	1	\N	dps	\N
836	TK	TK10	Horde	Rogue	3	6	48	16656	4192	394	1	\N	dps	\N
837	TK	TK10	Alliance	Druid	2	0	8	8264	2812	110	\N	1	dps	\N
838	TK	TK10	Horde	Shaman	0	2	49	5464	66158	546	1	\N	heal	\N
839	TK	TK10	Horde	Demon Hunter	12	4	47	28853	4672	395	1	\N	dps	\N
840	TK	TK10	Alliance	Warrior	2	7	25	8935	759	222	\N	1	dps	\N
841	TK	TK10	Alliance	Demon Hunter	5	7	27	44477	7646	226	\N	1	dps	\N
842	TK	TK10	Horde	Paladin	8	4	41	35243	8573	530	1	\N	dps	\N
843	TK	TK10	Horde	Priest	0	2	45	344	41371	538	1	\N	heal	\N
844	TK	TK10	Horde	Paladin	10	0	52	28796	5919	402	1	\N	dps	\N
845	TK	TK10	Alliance	Paladin	0	5	26	3389	19144	224	\N	1	heal	\N
846	TK	TK10	Alliance	Demon Hunter	2	7	20	21366	260	211	\N	1	dps	\N
847	TK	TK10	Horde	Warlock	2	6	42	22136	19597	382	1	\N	dps	\N
848	TK	TK10	Alliance	Warlock	8	4	29	28722	16078	231	\N	1	dps	\N
849	TK	TK11	Alliance	Warlock	3	6	15	25219	5384	145	\N	1	dps	\N
850	TK	TK11	Horde	Rogue	2	2	43	9855	1301	379	1	\N	dps	\N
851	TK	TK11	Alliance	Warlock	0	0	4	8131	5667	104	\N	1	dps	\N
852	TK	TK11	Horde	Death Knight	3	1	43	38620	7204	529	1	\N	dps	\N
853	TK	TK11	Alliance	Shaman	3	6	15	10415	2947	145	\N	1	dps	\N
854	TK	TK11	Horde	Demon Hunter	4	2	44	27838	4272	381	1	\N	dps	\N
855	TK	TK11	Horde	Warrior	5	4	35	16007	2137	513	1	\N	dps	\N
856	TK	TK11	Horde	Shaman	0	3	41	4601	42311	525	1	\N	heal	\N
857	TK	TK11	Alliance	Death Knight	2	4	15	22344	6085	146	\N	1	dps	\N
858	TK	TK11	Alliance	Death Knight	2	3	13	8995	5168	141	\N	1	dps	\N
859	TK	TK11	Alliance	Death Knight	4	4	16	28500	7693	146	\N	1	dps	\N
860	TK	TK11	Alliance	Rogue	4	5	14	14064	3306	143	\N	1	dps	\N
861	TK	TK11	Horde	Rogue	1	1	43	10390	1553	379	1	\N	dps	\N
862	TK	TK11	Horde	Paladin	16	0	45	35090	11090	383	1	\N	dps	\N
863	TK	TK11	Horde	Druid	2	5	40	10537	7122	373	1	\N	dps	\N
864	TK	TK11	Horde	Warrior	6	2	42	20952	2547	527	1	\N	dps	\N
865	TK	TK11	Alliance	Rogue	1	5	18	12065	5445	152	\N	1	dps	\N
866	TK	TK11	Alliance	Druid	0	3	18	861	24177	150	\N	1	heal	\N
867	TK	TK11	Horde	Shaman	6	0	45	13205	5719	533	1	\N	dps	\N
868	TK	TK11	Alliance	Demon Hunter	1	7	11	6843	828	137	\N	1	dps	\N
869	DG	DG2	Alliance	Hunter	11	2	40	108000	5220	288	\N	1	dps	\N
870	DG	DG2	Alliance	Priest	3	3	38	17345	114000	280	\N	1	heal	\N
871	DG	DG2	Horde	Mage	0	5	34	25932	2574	500	1	\N	dps	\N
872	DG	DG2	Alliance	Shaman	0	5	32	10381	84526	262	\N	1	heal	\N
873	DG	DG2	Horde	Demon Hunter	6	5	28	82896	19311	496	1	\N	dps	\N
874	DG	DG2	Horde	Death Knight	0	3	50	54630	40886	522	1	\N	dps	\N
875	DG	DG2	Alliance	Warlock	0	4	25	14252	2515	146	\N	1	dps	\N
876	DG	DG2	Alliance	Paladin	3	4	26	27830	18223	257	\N	1	dps	\N
877	DG	DG2	Horde	Rogue	3	5	28	34304	4673	500	1	\N	dps	\N
878	DG	DG2	Horde	Death Knight	13	3	53	90367	44423	534	1	\N	dps	\N
879	DG	DG2	Alliance	Mage	1	5	21	22558	8547	241	\N	1	dps	\N
880	DG	DG2	Alliance	Shaman	1	6	27	46366	4360	262	\N	1	dps	\N
881	DG	DG2	Horde	Shaman	0	5	46	14366	127000	524	1	\N	heal	\N
882	DG	DG2	Alliance	Paladin	1	5	34	75867	37212	269	\N	1	dps	\N
883	DG	DG2	Horde	Hunter	3	2	45	65943	7335	516	1	\N	dps	\N
884	DG	DG2	Alliance	Priest	8	3	35	86958	26805	278	\N	1	dps	\N
885	DG	DG2	Horde	Rogue	0	4	37	24030	1692	500	1	\N	dps	\N
886	DG	DG2	Alliance	Priest	0	3	30	2633	77562	262	\N	1	heal	\N
887	DG	DG2	Horde	Paladin	0	3	33	35984	49038	499	1	\N	heal	\N
888	DG	DG2	Horde	Mage	13	1	54	54259	20971	534	1	\N	dps	\N
889	DG	DG2	Horde	Priest	2	2	50	24136	116000	523	1	\N	heal	\N
890	DG	DG2	Alliance	Paladin	4	5	20	127000	19158	121	\N	1	dps	\N
891	DG	DG2	Alliance	Rogue	2	6	31	40234	3371	267	\N	1	dps	\N
892	DG	DG2	Alliance	Warrior	2	4	27	21209	8950	253	\N	1	dps	\N
893	DG	DG2	Horde	Warrior	5	6	24	37263	12972	498	1	\N	dps	\N
894	DG	DG2	Horde	Demon Hunter	4	4	52	53240	6766	536	1	\N	dps	\N
895	DG	DG2	Horde	Death Knight	3	2	45	57348	29749	522	1	\N	dps	\N
896	DG	DG2	Horde	Rogue	6	2	47	48958	19528	519	1	\N	dps	\N
897	DG	DG2	Alliance	Shaman	5	3	36	65684	7134	273	\N	1	dps	\N
898	DG	DG2	Alliance	Mage	10	3	40	104000	4278	288	\N	1	dps	\N
899	AB	AB1	Alliance	Mage	8	4	37	87559	13923	572	1	\N	dps	\N
900	AB	AB1	Horde	Paladin	0	0	7	103	2288	119	\N	1	heal	\N
901	AB	AB1	Horde	Hunter	1	7	11	74415	7212	205	\N	1	dps	\N
902	AB	AB1	Alliance	Warrior	12	2	39	80285	11346	816	1	\N	dps	\N
903	AB	AB1	Alliance	Druid	0	3	42	1368	175000	598	1	\N	heal	\N
904	AB	AB1	Horde	Mage	2	3	18	152000	4749	220	\N	1	dps	\N
905	AB	AB1	Horde	Priest	0	2	9	7749	83060	194	\N	1	heal	\N
906	AB	AB1	Alliance	Mage	2	2	53	64224	10864	856	1	\N	dps	\N
907	AB	AB1	Alliance	Monk	2	2	36	3490	187000	568	1	\N	heal	\N
908	AB	AB1	Horde	Shaman	0	7	16	20477	143000	215	\N	1	heal	\N
909	AB	AB1	Horde	Paladin	1	1	4	11769	3325	133	\N	1	dps	\N
910	AB	AB1	Alliance	Mage	2	0	51	111000	4314	847	1	\N	dps	\N
911	AB	AB1	Alliance	Druid	6	0	51	82081	11540	623	1	\N	dps	\N
912	AB	AB1	Alliance	Warrior	4	1	46	125000	18200	825	1	\N	dps	\N
913	AB	AB1	Horde	Monk	3	6	11	92806	35406	208	\N	1	dps	\N
914	AB	AB1	Horde	Mage	1	4	14	83602	23018	209	\N	1	dps	\N
915	AB	AB1	Horde	Druid	0	1	10	35208	17948	209	\N	1	dps	\N
916	AB	AB1	Alliance	Death Knight	4	1	44	186000	21857	594	1	\N	dps	\N
917	AB	AB1	Alliance	Warrior	6	4	25	66806	4860	369	1	\N	dps	\N
918	AB	AB1	Horde	Mage	0	3	9	20849	4178	130	\N	1	dps	\N
919	AB	AB1	Horde	Warrior	5	3	18	181000	37231	221	\N	1	dps	\N
920	AB	AB1	Horde	Warlock	4	5	13	119000	33349	206	\N	1	dps	\N
921	AB	AB1	Alliance	Druid	0	1	46	9630	168000	605	1	\N	heal	\N
922	AB	AB1	Horde	Rogue	2	5	17	49837	10297	214	\N	1	dps	\N
923	AB	AB1	Alliance	Demon Hunter	1	1	47	39138	21708	833	1	\N	dps	\N
924	AB	AB1	Alliance	Shaman	0	1	36	0	143000	563	1	\N	heal	\N
925	AB	AB1	Horde	Shaman	2	1	19	26670	120000	221	\N	1	heal	\N
926	AB	AB1	Horde	Priest	0	5	12	5332	225000	205	\N	1	heal	\N
927	AB	AB1	Alliance	Priest	2	2	42	214000	33147	587	1	\N	dps	\N
928	AB	AB1	Alliance	Rogue	3	3	24	34426	2802	543	1	\N	dps	\N
929	SM	SM1	Horde	Paladin	3	3	18	58874	29732	353	1	\N	dps	\N
930	SM	SM1	Horde	Warrior	1	4	19	33188	42355	356	1	\N	dps	\N
931	SM	SM1	Alliance	Druid	7	4	36	69535	4134	366	\N	1	dps	\N
932	SM	SM1	Horde	Paladin	4	5	18	60889	21554	504	1	\N	dps	\N
933	SM	SM1	Horde	Rogue	1	8	17	33717	1863	502	1	\N	dps	\N
934	SM	SM1	Alliance	Demon Hunter	14	2	45	92862	9878	385	\N	1	dps	\N
935	SM	SM1	Alliance	Death Knight	3	2	43	47029	24575	381	\N	1	dps	\N
936	SM	SM1	Horde	Shaman	2	6	16	6920	72906	499	1	\N	heal	\N
937	SM	SM1	Alliance	Paladin	4	4	42	97230	23473	379	\N	1	dps	\N
938	SM	SM1	Alliance	Rogue	9	1	41	54439	9108	387	\N	1	dps	\N
939	SM	SM1	Horde	Death Knight	1	8	13	38732	14942	493	1	\N	dps	\N
940	SM	SM1	Horde	Paladin	0	2	18	854	46905	503	1	\N	heal	\N
941	SM	SM1	Alliance	Warlock	5	0	45	77695	29075	381	\N	1	dps	\N
942	SM	SM1	Alliance	Mage	5	1	46	59585	10493	385	\N	1	dps	\N
943	SM	SM1	Alliance	Paladin	0	1	42	3569	31105	371	\N	1	heal	\N
944	SM	SM1	Alliance	Priest	0	3	39	0	56190	364	\N	1	heal	\N
945	SM	SM1	Horde	Mage	4	7	15	37121	8251	497	1	\N	dps	\N
946	SM	SM1	Horde	Shaman	2	4	14	27238	13950	346	1	\N	dps	\N
947	SM	SM1	Alliance	Rogue	4	2	43	39846	11276	373	\N	1	dps	\N
948	SM	SM1	Horde	Druid	1	4	10	14547	7167	349	1	\N	dps	\N
949	SM	SM2	Alliance	Warlock	2	1	9	19175	11495	171	\N	1	dps	\N
950	SM	SM2	Horde	Paladin	1	0	28	32778	16829	511	1	\N	dps	\N
951	SM	SM2	Alliance	Priest	1	3	11	9731	25164	175	\N	1	heal	\N
952	SM	SM2	Alliance	Druid	0	2	11	0	27090	175	\N	1	heal	\N
953	SM	SM2	Horde	Rogue	5	0	30	22297	6474	364	1	\N	dps	\N
954	SM	SM2	Horde	Shaman	0	0	29	4351	29830	512	1	\N	heal	\N
955	SM	SM2	Horde	Monk	6	3	27	21183	5864	358	1	\N	dps	\N
956	SM	SM2	Horde	Demon Hunter	4	2	22	34860	5997	498	1	\N	dps	\N
957	SM	SM2	Alliance	Druid	2	6	7	12566	3388	167	\N	1	dps	\N
958	SM	SM2	Alliance	Mage	0	3	10	8487	4184	173	\N	1	dps	\N
959	SM	SM2	Alliance	Monk	0	3	9	231	18752	172	\N	1	heal	\N
960	SM	SM2	Horde	Shaman	0	1	28	3344	22031	511	1	\N	heal	\N
961	SM	SM2	Alliance	Hunter	1	2	9	25034	2712	171	\N	1	dps	\N
962	SM	SM2	Horde	Demon Hunter	0	3	25	18449	1208	504	1	\N	dps	\N
963	SM	SM2	Horde	Mage	4	2	28	18103	3524	360	1	\N	dps	\N
964	SM	SM2	Alliance	Paladin	4	2	11	18293	5895	175	\N	1	dps	\N
965	SM	SM2	Alliance	Druid	1	6	5	18722	3017	163	\N	1	dps	\N
966	SM	SM2	Horde	Priest	2	0	30	24230	4600	515	1	\N	dps	\N
967	SM	SM2	Horde	Warrior	9	0	27	29834	1206	359	1	\N	dps	\N
968	SM	SM2	Alliance	Demon Hunter	0	3	8	7371	1094	170	\N	1	dps	\N
969	SM	SM3	Alliance	Monk	0	3	8	627	33250	179	\N	1	heal	\N
970	SM	SM3	Horde	Hunter	1	3	11	7047	1574	331	1	\N	dps	\N
971	SM	SM3	Alliance	Monk	0	1	6	10312	4178	174	\N	1	dps	\N
972	SM	SM3	Alliance	Paladin	2	2	9	15921	5745	180	\N	1	dps	\N
973	SM	SM3	Horde	Priest	2	0	22	16695	22637	503	1	\N	heal	\N
974	SM	SM3	Horde	Monk	3	0	22	15795	1885	503	1	\N	dps	\N
975	SM	SM3	Horde	Shaman	1	0	22	2259	23537	503	1	\N	heal	\N
976	SM	SM3	Horde	Warlock	2	1	21	24101	6103	501	1	\N	dps	\N
977	SM	SM3	Horde	Rogue	2	3	19	12530	1544	497	1	\N	dps	\N
978	SM	SM3	Horde	Priest	1	1	22	9411	15502	503	1	\N	heal	\N
979	SM	SM3	Alliance	Warrior	2	1	10	11506	2010	183	\N	1	dps	\N
980	SM	SM3	Alliance	Monk	0	2	8	11	44107	178	\N	1	heal	\N
981	SM	SM3	Horde	Druid	1	1	21	31688	3559	351	1	\N	dps	\N
982	SM	SM3	Alliance	Paladin	1	3	8	25819	4506	178	\N	1	dps	\N
983	SM	SM3	Alliance	Paladin	0	4	9	550	21572	180	\N	1	heal	\N
984	SM	SM3	Alliance	Warlock	2	3	8	22348	8949	179	\N	1	dps	\N
985	SM	SM3	Alliance	Warlock	0	1	9	2344	1212	181	\N	1	dps	\N
986	SM	SM3	Alliance	Rogue	2	2	9	20121	5539	181	\N	1	dps	\N
987	SM	SM3	Horde	Demon Hunter	5	0	22	51111	1815	353	1	\N	dps	\N
988	SM	SM3	Horde	Warrior	4	1	17	13033	2667	493	1	\N	dps	\N
989	SM	SM4	Alliance	Hunter	0	1	3	10996	2777	46	\N	1	dps	\N
990	SM	SM4	Horde	Druid	0	0	29	2655	50557	522	1	\N	heal	\N
991	SM	SM4	Alliance	Death Knight	1	3	3	16115	8601	131	\N	1	dps	\N
992	SM	SM4	Horde	Shaman	0	1	21	3084	16988	507	1	\N	heal	\N
993	SM	SM4	Horde	Warlock	2	0	32	25122	20017	531	1	\N	dps	\N
994	SM	SM4	Alliance	Priest	0	4	4	758	69932	133	\N	1	heal	\N
995	SM	SM4	Alliance	Hunter	0	6	3	17776	4451	131	\N	1	dps	\N
996	SM	SM4	Alliance	Hunter	0	4	2	24289	7469	129	\N	1	dps	\N
997	SM	SM4	Alliance	Demon Hunter	1	4	2	26889	2441	129	\N	1	dps	\N
998	SM	SM4	Horde	Hunter	2	0	26	14930	0	514	1	\N	dps	\N
999	SM	SM4	Alliance	Druid	0	2	4	16146	10727	133	\N	1	dps	\N
1000	SM	SM4	Horde	Paladin	3	0	26	23179	10309	364	1	\N	dps	\N
1001	SM	SM4	Alliance	Paladin	0	2	4	532	18877	133	\N	1	heal	\N
1002	SM	SM4	Alliance	Druid	1	2	4	19423	3462	133	\N	1	dps	\N
1003	SM	SM4	Horde	Hunter	8	0	31	40339	3033	519	1	\N	dps	\N
1004	SM	SM4	Horde	Paladin	2	1	25	38413	4034	515	1	\N	dps	\N
1005	SM	SM4	Horde	Warlock	2	1	23	7834	7995	511	1	\N	dps	\N
1006	SM	SM4	Alliance	Hunter	1	3	4	13804	4858	133	\N	1	dps	\N
1007	SM	SM4	Horde	Shaman	5	1	25	41502	15860	513	1	\N	dps	\N
1008	SM	SM4	Horde	Warrior	7	0	28	27138	4993	520	1	\N	dps	\N
1009	SM	SM5	Alliance	Paladin	0	7	17	25707	8513	353	\N	1	dps	1
1010	SM	SM5	Alliance	Demon Hunter	6	2	23	52817	6649	374	\N	1	dps	1
1011	SM	SM5	Horde	Mage	0	3	42	28290	3489	454	1	\N	dps	1
1012	SM	SM5	Horde	Priest	9	2	44	67718	29004	458	1	\N	dps	1
1013	SM	SM5	Alliance	Demon Hunter	0	5	17	16550	2846	354	\N	1	dps	1
1014	SM	SM5	Horde	Rogue	5	1	45	25108	2475	460	1	\N	dps	1
1015	SM	SM5	Horde	Warlock	8	2	41	39582	32892	452	1	\N	dps	1
1016	SM	SM5	Alliance	Death Knight	0	6	18	18019	19339	355	\N	1	dps	1
1017	SM	SM5	Horde	Demon Hunter	3	0	43	31192	11082	681	1	\N	dps	1
1018	SM	SM5	Horde	Shaman	1	6	42	6339	59138	454	1	\N	heal	1
1019	SM	SM5	Alliance	Mage	4	3	22	22585	8329	373	\N	1	dps	1
1020	SM	SM5	Alliance	Druid	0	4	18	200	49004	357	\N	1	heal	1
1021	SM	SM5	Alliance	Hunter	6	4	21	74614	3830	367	\N	1	dps	1
1022	SM	SM5	Horde	Priest	10	1	45	65592	42314	460	1	\N	dps	1
1023	SM	SM5	Horde	Priest	4	2	38	46117	34324	671	1	\N	dps	1
1024	SM	SM5	Alliance	Shaman	0	8	13	43728	222	336	\N	1	dps	1
1025	SM	SM5	Horde	Shaman	4	4	37	38664	15802	444	1	\N	dps	1
1026	SM	SM5	Alliance	Paladin	1	2	14	17248	7764	340	\N	1	dps	1
1027	SM	SM5	Alliance	Warrior	8	5	21	75974	18398	366	\N	1	dps	1
1028	SM	SM5	Horde	Shaman	1	4	44	24701	10861	683	1	\N	dps	1
1029	SM	SM6	Alliance	Demon Hunter	5	0	45	97774	9828	1038	1	\N	dps	1
1030	SM	SM6	Alliance	Mage	9	2	43	101000	7752	697	1	\N	dps	1
1031	SM	SM6	Alliance	Druid	9	0	43	46814	7998	1033	1	\N	dps	1
1032	SM	SM6	Alliance	Demon Hunter	2	3	38	20282	23129	674	1	\N	dps	1
1033	SM	SM6	Horde	Rogue	0	2	14	23622	15360	267	\N	1	dps	1
1034	SM	SM6	Horde	Monk	1	5	13	13080	91131	265	\N	1	heal	1
1035	SM	SM6	Alliance	Mage	1	3	41	33715	4679	1029	1	\N	dps	1
1036	SM	SM6	Alliance	Hunter	5	5	39	62260	12390	686	1	\N	dps	1
1037	SM	SM6	Horde	Druid	2	5	15	76711	17624	269	\N	1	dps	1
1038	SM	SM6	Alliance	Shaman	2	3	33	10932	106000	1009	1	\N	heal	1
1039	SM	SM6	Horde	Rogue	0	3	18	42584	11807	277	\N	1	dps	1
1040	SM	SM6	Horde	Hunter	1	5	13	34785	6793	265	\N	1	dps	1
1041	SM	SM6	Alliance	Druid	1	1	44	5553	166000	697	1	\N	heal	1
1042	SM	SM6	Alliance	Hunter	6	0	45	86061	9634	697	1	\N	dps	1
1043	SM	SM6	Horde	Death Knight	3	5	14	61492	38798	267	\N	1	dps	1
1044	SM	SM6	Horde	Mage	6	5	13	85944	9637	265	\N	1	dps	1
1045	SM	SM6	Horde	Priest	0	4	12	20109	92959	263	\N	1	heal	1
1046	SM	SM6	Horde	Warrior	4	9	14	71378	7362	269	\N	1	dps	1
1047	SM	SM6	Alliance	Warrior	6	1	43	51626	10665	1031	1	\N	dps	1
1048	SM	SM6	Horde	Warlock	1	3	15	35339	7279	271	\N	1	dps	1
1049	TP	TP1	Horde	Mage	0	1	17	22143	3281	145	\N	1	dps	\N
1050	TP	TP1	Horde	Mage	1	0	3	7494	3247	99	\N	1	dps	\N
1051	TP	TP1	Alliance	Demon Hunter	2	2	18	38970	7722	771	1	\N	dps	\N
1052	TP	TP1	Horde	Hunter	2	1	17	17714	1851	135	\N	1	dps	\N
1053	TP	TP1	Horde	Shaman	1	3	12	8477	44467	134	\N	1	heal	\N
1054	TP	TP1	Horde	Mage	6	2	17	36622	3478	135	\N	1	dps	\N
1055	TP	TP1	Alliance	Warrior	2	3	14	25597	4139	532	1	\N	dps	\N
1056	TP	TP1	Alliance	Priest	0	0	19	6337	56269	771	1	\N	heal	\N
1057	TP	TP1	Alliance	Priest	2	2	19	18235	6863	557	1	\N	dps	\N
1058	TP	TP1	Alliance	Demon Hunter	0	1	17	13306	974	765	1	\N	dps	\N
1059	TP	TP1	Alliance	Warrior	4	2	16	10912	8847	538	1	\N	dps	\N
1060	TP	TP1	Alliance	Priest	3	3	15	45563	18001	748	1	\N	dps	\N
1061	TP	TP1	Horde	Paladin	2	3	11	50249	7529	125	\N	1	dps	\N
1062	TP	TP1	Alliance	Druid	0	1	11	418	18912	751	1	\N	heal	\N
1063	TP	TP1	Alliance	Priest	6	3	16	62836	16103	542	1	\N	dps	\N
1064	TP	TP1	Horde	Mage	2	4	14	32945	7101	138	\N	1	dps	\N
1065	TP	TP1	Alliance	Warrior	3	3	20	27658	9190	762	1	\N	dps	\N
1066	TP	TP1	Horde	Warrior	2	3	11	30239	8984	134	\N	1	dps	\N
1067	TP	TP1	Horde	Warlock	1	2	17	26684	13666	145	\N	1	dps	\N
1068	TP	TP1	Horde	Priest	0	2	18	1698	46505	154	\N	1	heal	\N
1069	TP	TP2	Alliance	Druid	0	0	16	4830	55789	229	\N	1	heal	\N
1070	TP	TP2	Alliance	Rogue	0	4	5	21946	2036	198	\N	1	dps	\N
1071	TP	TP2	Horde	Demon Hunter	1	1	16	23722	7194	374	1	\N	dps	\N
1072	TP	TP2	Alliance	Demon Hunter	0	3	14	25126	304	225	\N	1	dps	\N
1073	TP	TP2	Alliance	Demon Hunter	3	3	16	47351	5682	229	\N	1	dps	\N
1074	TP	TP2	Horde	Warrior	5	4	17	41586	7895	525	1	\N	dps	\N
1075	TP	TP2	Horde	Death Knight	4	5	15	72716	33793	524	1	\N	dps	\N
1076	TP	TP2	Alliance	Hunter	6	1	18	63642	5939	240	\N	1	dps	\N
1077	TP	TP2	Horde	Shaman	1	3	16	3506	34528	519	1	\N	heal	\N
1078	TP	TP2	Horde	Shaman	1	0	20	3621	51821	533	1	\N	heal	\N
1079	TP	TP2	Alliance	Mage	2	3	15	8895	3802	227	\N	1	dps	\N
1080	TP	TP2	Horde	Hunter	2	0	17	29710	1346	522	1	\N	dps	\N
1081	TP	TP2	Horde	Monk	3	2	13	37836	12401	512	1	\N	dps	\N
1082	TP	TP2	Alliance	Mage	2	3	16	43180	6972	233	\N	1	dps	\N
1083	TP	TP2	Horde	Hunter	4	1	14	56188	8653	521	1	\N	dps	\N
1084	TP	TP2	Alliance	Priest	0	1	18	9967	101000	240	\N	1	heal	\N
1085	TP	TP2	Alliance	Shaman	5	1	16	34760	9172	229	\N	1	dps	\N
1086	TP	TP2	Horde	Monk	2	0	19	23447	10391	381	1	\N	dps	\N
1087	TP	TP2	Horde	Warlock	2	2	16	23480	8633	522	1	\N	dps	\N
1088	TP	TP2	Alliance	Warlock	0	6	5	25204	12891	200	\N	1	dps	\N
1089	TP	TP3	Alliance	Hunter	2	3	35	47816	3734	588	1	\N	dps	\N
1090	TP	TP3	Horde	Hunter	2	4	11	61727	7806	143	\N	1	dps	\N
1091	TP	TP3	Alliance	Shaman	7	3	44	63438	12212	840	1	\N	dps	\N
1092	TP	TP3	Alliance	Paladin	3	2	44	29177	13799	838	1	\N	dps	\N
1093	TP	TP3	Horde	Druid	1	5	12	58505	13698	149	\N	1	dps	\N
1094	TP	TP3	Horde	Hunter	0	1	9	28690	3583	130	\N	1	dps	\N
1095	TP	TP3	Horde	Druid	0	3	3	31933	553	101	\N	1	dps	\N
1096	TP	TP3	Alliance	Druid	1	0	48	4115	112000	851	1	\N	heal	\N
1097	TP	TP3	Horde	Shaman	0	6	14	13297	75994	150	\N	1	heal	\N
1098	TP	TP3	Alliance	Hunter	9	3	34	49790	13549	819	1	\N	dps	\N
1099	TP	TP3	Alliance	Warrior	9	0	44	57203	11294	855	1	\N	dps	\N
1100	TP	TP3	Alliance	Demon Hunter	6	1	44	49709	8170	612	1	\N	dps	\N
1101	TP	TP3	Alliance	Demon Hunter	4	1	47	53810	18656	846	1	\N	dps	\N
1102	TP	TP3	Horde	Paladin	0	2	10	1832	27101	142	\N	1	heal	\N
1103	TP	TP3	Alliance	Hunter	7	2	41	68436	6207	831	1	\N	dps	\N
1104	TP	TP3	Horde	Death Knight	4	6	13	62293	32154	147	\N	1	dps	\N
1105	TP	TP3	Horde	Hunter	0	8	12	28793	8016	145	\N	1	dps	\N
1106	TP	TP3	Horde	Rogue	6	5	13	30749	7144	152	\N	1	dps	\N
1107	TP	TP3	Alliance	Paladin	0	0	39	11367	73497	601	1	\N	heal	\N
1108	TP	TP3	Horde	Warrior	1	7	13	41640	8967	148	\N	1	dps	\N
1109	TP	TP4	Horde	Priest	1	4	8	13921	87910	130	\N	1	heal	\N
1110	TP	TP4	Alliance	Druid	3	0	28	50860	7164	798	1	\N	dps	\N
1111	TP	TP4	Alliance	Paladin	0	1	21	15657	2153	769	1	\N	dps	\N
1112	TP	TP4	Alliance	Monk	0	3	23	0	52110	783	1	\N	heal	\N
1113	TP	TP4	Alliance	Hunter	4	0	29	61922	4291	800	1	\N	dps	\N
1114	TP	TP4	Alliance	Priest	1	1	19	6425	29085	536	1	\N	heal	\N
1115	TP	TP4	Horde	Rogue	1	2	8	14917	3476	119	\N	1	dps	\N
1116	TP	TP4	Horde	Shaman	0	5	5	3842	62007	123	\N	1	heal	\N
1117	TP	TP4	Alliance	Rogue	2	3	20	18830	3017	779	1	\N	dps	\N
1118	TP	TP4	Horde	Death Knight	0	4	7	36793	19860	117	\N	1	dps	\N
1119	TP	TP4	Alliance	Rogue	6	1	28	50888	9844	797	1	\N	dps	\N
1120	TP	TP4	Alliance	Demon Hunter	6	0	19	29894	4117	779	1	\N	dps	\N
1121	TP	TP4	Horde	Mage	0	2	6	15570	4170	116	\N	1	dps	\N
1122	TP	TP4	Horde	Demon Hunter	2	3	9	32772	1879	122	\N	1	dps	\N
1123	TP	TP4	Alliance	Death Knight	4	0	26	61077	24808	788	1	\N	dps	\N
1124	TP	TP4	Horde	Demon Hunter	0	0	0	5001	733	75	\N	1	dps	\N
1125	TP	TP4	Alliance	Death Knight	6	0	26	61517	9761	788	1	\N	dps	\N
1126	TP	TP4	Horde	Rogue	2	5	8	15955	2241	120	\N	1	dps	\N
1127	TP	TP5	Horde	Monk	7	4	20	4700	1538	223	\N	1	dps	\N
1128	TP	TP5	Horde	Druid	2	2	20	16909	3091	221	\N	1	dps	\N
1129	TP	TP5	Alliance	Warrior	1	4	23	9736	281	511	1	\N	dps	\N
1130	TP	TP5	Alliance	Druid	0	2	25	27529	15028	365	1	\N	dps	\N
1131	TP	TP5	Horde	Mage	0	3	15	18951	6043	214	\N	1	dps	\N
1132	TP	TP5	Horde	Warrior	0	2	24	23575	14533	229	\N	1	dps	\N
1133	TP	TP5	Horde	Warrior	3	7	20	28060	3540	225	\N	1	dps	\N
1134	TP	TP5	Alliance	Monk	3	2	22	19528	3417	363	1	\N	dps	\N
1135	TP	TP5	Alliance	Druid	1	0	22	6670	59085	510	1	\N	heal	\N
1136	TP	TP5	Alliance	Druid	2	2	21	1810	45132	509	1	\N	heal	\N
1137	TP	TP5	Horde	Shaman	0	3	23	7340	43717	226	\N	1	heal	\N
1138	TP	TP5	Horde	Shaman	9	5	24	47707	1239	234	\N	1	dps	\N
1139	TP	TP5	Horde	Hunter	5	3	26	27884	2194	234	\N	1	dps	\N
1140	TP	TP5	Alliance	Mage	9	1	28	42848	6232	378	1	\N	dps	\N
1141	TP	TP5	Horde	Warrior	1	2	22	25296	18631	226	\N	1	dps	\N
1142	TP	TP5	Alliance	Warrior	9	5	23	41415	2000	516	1	\N	dps	\N
1143	TP	TP5	Alliance	Monk	3	4	22	25214	7651	511	1	\N	dps	\N
1144	TP	TP5	Horde	Shaman	1	3	26	6652	60447	234	\N	1	heal	\N
1145	TP	TP5	Alliance	Death Knight	4	5	25	48198	12318	367	1	\N	dps	\N
1146	TP	TP5	Alliance	Demon Hunter	2	4	19	21789	3138	364	1	\N	dps	\N
1147	BG	BG17	Horde	Rogue	5	4	45	51237	17310	276	\N	1	dps	\N
1148	BG	BG17	Alliance	Hunter	5	4	31	73418	9144	800	1	\N	dps	\N
1149	BG	BG17	Alliance	Hunter	4	3	29	53194	10249	579	1	\N	dps	\N
1150	BG	BG17	Horde	Paladin	8	7	42	83612	32570	271	\N	1	dps	\N
1151	BG	BG17	Alliance	Mage	0	7	14	36666	19184	748	1	\N	dps	\N
1152	BG	BG17	Horde	Priest	4	2	13	24047	7801	215	\N	1	dps	\N
1153	BG	BG17	Alliance	Mage	7	6	30	77737	11825	572	1	\N	dps	\N
1154	BG	BG17	Horde	Shaman	0	6	43	20857	190000	271	\N	1	heal	\N
1155	BG	BG17	Horde	Demon Hunter	3	3	17	21081	2176	223	\N	1	dps	\N
1156	BG	BG17	Horde	Hunter	7	4	37	82055	11900	262	\N	1	dps	\N
1157	BG	BG17	Alliance	Hunter	7	4	37	82055	11900	594	1	\N	dps	\N
1158	BG	BG17	Alliance	Hunter	6	6	40	106000	15191	620	1	\N	dps	\N
1159	BG	BG17	Alliance	Death Knight	10	8	34	113000	40541	589	1	\N	dps	\N
1160	BG	BG17	Horde	Death Knight	5	3	45	43361	56030	278	\N	1	dps	\N
1161	BG	BG17	Alliance	Druid	0	3	36	1112	131000	821	1	\N	heal	\N
1162	BG	BG17	Horde	Shaman	7	4	46	110000	18815	278	\N	1	dps	\N
1163	BG	BG17	Alliance	Druid	1	3	12	12491	776	522	1	\N	dps	\N
1164	BG	BG17	Horde	Rogue	6	6	29	48068	12530	162	\N	1	dps	\N
1165	BG	BG17	Horde	Hunter	3	5	44	50097	12874	276	\N	1	dps	\N
1166	BG	BG17	Alliance	Demon Hunter	4	4	31	32712	23882	801	1	\N	dps	\N
1167	BG	BG18	Alliance	Demon Hunter	0	2	7	19628	776	181	\N	1	dps	\N
1168	BG	BG18	Horde	Paladin	2	1	12	23856	7983	345	1	\N	dps	\N
1169	BG	BG18	Horde	Death Knight	1	0	12	31701	33958	345	1	\N	dps	\N
1170	BG	BG18	Horde	Demon Hunter	3	2	6	33274	4903	340	1	\N	dps	\N
1171	BG	BG18	Horde	Priest	5	0	14	31383	8944	501	1	\N	dps	\N
1172	BG	BG18	Alliance	Warrior	0	2	4	19395	2871	170	\N	1	dps	\N
1173	BG	BG18	Horde	Warlock	3	1	12	40418	15685	495	1	\N	dps	\N
1174	BG	BG18	Alliance	Warrior	0	2	5	18057	3781	173	\N	1	dps	\N
1175	BG	BG18	Horde	Shaman	2	2	9	9390	46838	344	1	\N	heal	\N
1176	BG	BG18	Horde	Priest	0	1	12	5789	41588	495	1	\N	heal	\N
1177	BG	BG18	Horde	Warrior	3	1	7	23788	22274	346	1	\N	dps	\N
1178	BG	BG18	Alliance	Shaman	1	2	6	49148	4281	177	\N	1	dps	\N
1179	BG	BG18	Horde	Warlock	0	0	11	5652	5239	494	1	\N	dps	\N
1180	BG	BG18	Alliance	Monk	0	2	6	1864	61595	177	\N	1	heal	\N
1181	BG	BG18	Alliance	Paladin	0	3	4	7710	5657	170	\N	1	dps	\N
1182	BG	BG18	Alliance	Demon Hunter	2	3	4	47762	6200	170	\N	1	dps	\N
1183	BG	BG18	Horde	Rogue	2	0	15	10512	1988	364	1	\N	dps	\N
1184	BG	BG18	Alliance	Warrior	3	2	6	41592	7035	177	\N	1	dps	\N
1185	BG	BG18	Alliance	Hunter	1	1	7	14512	0	181	\N	1	dps	\N
1186	ES	ES1	Alliance	Druid	3	3	51	41345	6846	455	\N	1	dps	1
1187	ES	ES1	Alliance	Monk	1	5	47	4151	96845	439	\N	1	heal	1
1188	ES	ES1	Horde	Paladin	6	5	32	71658	18317	771	1	\N	dps	1
1189	ES	ES1	Alliance	Druid	0	0	65	10704	102000	475	\N	1	heal	1
1190	ES	ES1	Alliance	Paladin	6	3	55	54924	19525	457	\N	1	dps	1
1191	ES	ES1	Alliance	Hunter	8	3	58	80393	9125	461	\N	1	dps	1
1192	ES	ES1	Alliance	Hunter	4	4	55	42148	7130	454	\N	1	dps	1
1193	ES	ES1	Alliance	Priest	3	7	48	38369	16447	445	\N	1	dps	1
1194	ES	ES1	Horde	Shaman	0	5	37	7373	62941	775	1	\N	heal	1
1195	ES	ES1	Alliance	Paladin	4	2	61	52221	9759	297	\N	1	dps	1
1196	ES	ES1	Alliance	Warrior	3	2	53	23774	3855	473	\N	1	dps	1
1197	ES	ES1	Horde	Druid	0	4	36	3437	126000	547	1	\N	heal	1
1198	ES	ES1	Alliance	Druid	1	3	25	15781	4953	325	\N	1	dps	1
1199	ES	ES1	Horde	Mage	8	5	29	79798	8942	543	1	\N	dps	1
1200	ES	ES1	Horde	Warlock	2	3	28	84128	15132	537	1	\N	dps	1
1201	ES	ES1	Alliance	Paladin	9	1	66	119000	26688	481	\N	1	dps	1
1202	ES	ES1	Alliance	Druid	7	1	60	52809	11500	293	\N	1	dps	1
1203	ES	ES1	Horde	Demon Hunter	4	5	24	67262	6141	758	1	\N	dps	1
1204	ES	ES1	Alliance	Paladin	9	1	63	170000	62122	470	\N	1	dps	1
1205	ES	ES1	Horde	Demon Hunter	1	2	23	23140	4461	536	1	\N	dps	1
1206	ES	ES1	Horde	Death Knight	6	6	30	68159	24001	541	1	\N	dps	1
1207	ES	ES1	Horde	Warlock	3	0	33	26812	8050	545	1	\N	dps	1
1208	ES	ES1	Horde	Druid	2	6	32	61954	3716	766	1	\N	dps	1
1209	ES	ES1	Horde	Death Knight	2	3	34	58974	18277	544	1	\N	dps	1
1210	ES	ES1	Alliance	Druid	0	3	54	301	158000	453	\N	1	heal	1
1211	ES	ES1	Horde	Demon Hunter	3	5	29	80266	21357	542	1	\N	dps	1
1212	ES	ES1	Horde	Monk	1	6	33	1705	116000	769	1	\N	heal	1
1213	ES	ES1	Horde	Demon Hunter	2	3	28	45502	12907	550	1	\N	dps	1
1214	ES	ES1	Alliance	Death Knight	5	5	46	53833	16665	440	\N	1	dps	1
1215	ES	ES2	Horde	Shaman	1	0	20	4200	20294	399	1	\N	heal	1
1216	ES	ES2	Horde	Demon Hunter	4	0	43	27521	11046	554	1	\N	dps	1
1217	ES	ES2	Horde	Rogue	3	0	38	13482	6591	542	1	\N	dps	1
1218	ES	ES2	Alliance	Shaman	0	4	1	11107	3005	234	\N	1	dps	1
1219	ES	ES2	Alliance	Hunter	0	3	0	30026	6214	229	\N	1	dps	1
1220	ES	ES2	Alliance	Druid	1	5	1	34245	6072	231	\N	1	dps	1
1221	ES	ES2	Horde	Mage	4	0	33	34665	4104	559	1	\N	dps	1
1222	ES	ES2	Alliance	Warlock	0	4	1	28212	12558	233	\N	1	dps	1
1223	ES	ES2	Horde	Paladin	5	1	27	37387	14109	539	1	\N	dps	1
1224	ES	ES2	Horde	Hunter	1	1	37	8275	1473	547	1	\N	dps	1
1225	ES	ES2	Horde	Shaman	0	1	35	9548	44439	551	1	\N	heal	1
1226	ES	ES2	Alliance	Warrior	1	4	2	33650	7070	237	\N	1	dps	1
1227	ES	ES2	Alliance	Rogue	0	5	2	13750	5845	235	\N	1	dps	1
1228	ES	ES2	Horde	Rogue	5	0	27	32895	7786	547	1	\N	dps	1
1229	ES	ES2	Alliance	Druid	0	3	1	13175	6591	233	\N	1	dps	1
1230	ES	ES2	Horde	Priest	5	0	40	60040	15011	778	1	\N	dps	1
1231	ES	ES2	Alliance	Rogue	0	2	3	28351	12590	239	\N	1	dps	1
1232	ES	ES2	Horde	Rogue	1	0	23	3668	910	533	1	\N	dps	1
1233	ES	ES2	Alliance	Druid	0	6	1	0	22660	231	\N	1	heal	1
1234	ES	ES2	Horde	Warlock	7	0	43	39438	16546	551	1	\N	dps	1
1235	ES	ES2	Horde	Shaman	5	0	35	31776	6985	782	1	\N	dps	1
1236	ES	ES2	Alliance	Mage	0	5	1	14947	1264	231	\N	1	dps	1
1237	ES	ES2	Horde	Death Knight	6	0	37	37951	13729	779	1	\N	dps	1
1238	ES	ES2	Horde	Death Knight	5	0	44	40162	38360	560	1	\N	dps	1
1239	ES	ES2	Alliance	Warrior	1	3	2	18236	669	235	\N	1	dps	1
1240	ES	ES2	Horde	Death Knight	0	0	44	30676	2575	570	1	\N	dps	1
1241	ES	ES2	Alliance	Druid	0	4	1	702	43364	231	\N	1	heal	1
1242	ES	ES3	Horde	Shaman	4	4	23	87355	11429	158	\N	1	dps	\N
1243	ES	ES3	Alliance	Shaman	0	0	30	9147	83387	533	1	\N	heal	\N
1244	ES	ES3	Alliance	Hunter	3	0	28	31257	3852	978	1	\N	dps	\N
1245	ES	ES3	Horde	Priest	6	0	25	79258	15808	141	\N	1	dps	\N
1246	ES	ES3	Horde	Warrior	2	5	21	54041	5374	159	\N	1	dps	\N
1247	ES	ES3	Horde	Monk	0	2	28	2167	120000	164	\N	1	heal	\N
1248	ES	ES3	Alliance	Hunter	4	4	34	47561	7910	980	1	\N	dps	\N
1249	ES	ES3	Horde	Hunter	0	2	12	6894	3059	147	\N	1	dps	\N
1250	ES	ES3	Horde	Paladin	1	2	19	56580	12849	155	\N	1	dps	\N
1251	ES	ES3	Alliance	Druid	0	2	39	0	107000	763	1	\N	heal	\N
1252	ES	ES3	Horde	Shaman	0	5	29	14677	93200	171	\N	1	heal	\N
1253	ES	ES3	Alliance	Death Knight	1	0	31	33700	6223	983	1	\N	dps	\N
1411	WG	WG18	Horde	Paladin	0	2	4	436	25396	325	1	\N	heal	\N
1254	ES	ES3	Alliance	Death Knight	3	3	36	73649	20787	982	1	\N	dps	\N
1255	ES	ES3	Horde	Demon Hunter	2	4	29	96061	5560	169	\N	1	dps	\N
1256	ES	ES3	Alliance	Demon Hunter	10	0	42	75412	14354	993	1	\N	dps	\N
1257	ES	ES3	Horde	Druid	1	1	28	39814	7010	172	\N	1	dps	\N
1258	ES	ES3	Horde	Rogue	2	1	28	30362	9186	165	\N	1	dps	\N
1259	ES	ES3	Alliance	Demon Hunter	2	0	38	37383	2751	1001	1	\N	dps	\N
1260	ES	ES3	Alliance	Demon Hunter	8	1	43	57601	13853	1007	1	\N	dps	\N
1261	ES	ES3	Horde	Demon Hunter	1	3	27	25643	2055	167	\N	1	dps	\N
1262	ES	ES3	Alliance	Shaman	2	5	30	36510	9813	764	1	\N	dps	\N
1263	ES	ES3	Alliance	Warrior	6	4	42	83743	12146	771	1	\N	dps	\N
1264	ES	ES3	Horde	Monk	0	3	28	1155	113000	166	\N	1	heal	\N
1265	ES	ES3	Alliance	Warrior	2	4	41	38146	10200	770	1	\N	dps	\N
1266	ES	ES3	Horde	Mage	5	3	24	48825	1987	164	\N	1	dps	\N
1267	ES	ES3	Horde	Death Knight	2	5	26	30744	13375	164	\N	1	dps	\N
1268	ES	ES3	Alliance	Warrior	5	3	33	57637	15991	976	1	\N	dps	\N
1269	ES	ES3	Alliance	Shaman	1	1	40	5477	122000	1002	1	\N	heal	\N
1270	ES	ES3	Alliance	Demon Hunter	1	3	42	73138	3303	769	1	\N	dps	\N
1271	ES	ES3	Horde	Rogue	2	7	22	43417	4718	158	\N	1	dps	\N
1272	SA	SA1	Alliance	Warrior	1	1	29	31129	12358	291	\N	1	dps	\N
1273	SA	SA1	Alliance	Warrior	2	6	33	57950	5784	305	\N	1	dps	\N
1274	SA	SA1	Horde	Warlock	1	6	27	79610	31523	339	1	\N	dps	\N
1275	SA	SA1	Horde	Priest	3	1	30	31158	11969	369	1	\N	dps	\N
1276	SA	SA1	Alliance	Paladin	0	0	5	18558	2724	162	\N	1	dps	\N
1277	SA	SA1	Alliance	Demon Hunter	2	4	27	62836	3631	292	\N	1	dps	\N
1278	SA	SA1	Horde	Hunter	2	4	30	88075	4727	353	1	\N	dps	\N
1279	SA	SA1	Horde	Druid	3	0	32	67420	17183	517	1	\N	dps	\N
1280	SA	SA1	Horde	Shaman	0	1	30	20887	87785	515	1	\N	heal	\N
1281	SA	SA1	Alliance	Priest	14	0	40	134000	39508	319	\N	1	dps	\N
1282	SA	SA1	Alliance	Priest	0	4	33	9542	138000	297	\N	1	heal	\N
1283	SA	SA1	Horde	Druid	1	1	28	6112	127000	352	1	\N	heal	\N
1284	SA	SA1	Horde	Hunter	3	1	28	102000	7947	357	1	\N	dps	\N
1285	SA	SA1	Alliance	Druid	1	2	29	56692	43818	298	\N	1	dps	\N
1286	SA	SA1	Alliance	Druid	6	1	40	100000	15722	314	\N	1	dps	\N
1287	SA	SA1	Alliance	Mage	1	2	23	69139	11493	282	\N	1	dps	\N
1288	SA	SA1	Alliance	Warrior	1	5	31	44539	27531	295	\N	1	dps	\N
1289	SA	SA1	Horde	Paladin	1	2	30	9302	128000	367	1	\N	heal	\N
1290	SA	SA1	Alliance	Monk	2	0	38	6712	140000	315	\N	1	heal	\N
1291	SA	SA1	Alliance	Shaman	2	3	24	12549	3227	278	\N	1	dps	\N
1292	SA	SA1	Alliance	Hunter	2	0	5	17682	376	157	\N	1	dps	\N
1293	SA	SA1	Horde	Shaman	1	2	33	44022	12723	354	1	\N	dps	\N
1294	SA	SA1	Alliance	Mage	3	3	35	107000	4168	299	\N	1	dps	\N
1295	SA	SA1	Horde	Shaman	2	3	20	52338	23003	494	1	\N	dps	\N
1296	SA	SA1	Alliance	Rogue	6	1	36	43844	14130	308	\N	1	dps	\N
1297	SA	SA1	Horde	Druid	4	4	31	61434	6308	513	1	\N	dps	\N
1298	SA	SA1	Horde	Shaman	2	8	23	69811	18597	494	1	\N	dps	\N
1299	SA	SA1	Horde	Monk	3	4	29	63178	26239	505	1	\N	dps	\N
1300	SA	SA1	Horde	Hunter	11	2	31	118000	13350	506	1	\N	dps	\N
1301	SA	SA1	Horde	Druid	0	3	32	3493	79902	514	1	\N	heal	\N
1302	SA	SA2	Alliance	Druid	0	4	26	9235	126000	376	\N	1	heal	1
1303	SA	SA2	Alliance	Warlock	2	1	32	38235	18496	396	\N	1	dps	1
1304	SA	SA2	Alliance	Hunter	4	5	27	77704	9750	367	\N	1	dps	1
1305	SA	SA2	Horde	Druid	0	2	46	6421	190000	441	1	\N	heal	1
1306	SA	SA2	Alliance	Paladin	0	3	27	36682	5253	383	\N	1	dps	1
1307	SA	SA2	Horde	Warlock	1	3	30	73297	36014	434	1	\N	dps	1
1308	SA	SA2	Alliance	Paladin	5	5	27	110000	27003	381	\N	1	dps	1
1309	SA	SA2	Horde	Hunter	7	1	48	188000	8090	443	1	\N	dps	1
1310	SA	SA2	Alliance	Shaman	0	3	27	6350	76122	384	\N	1	heal	1
1311	SA	SA2	Alliance	Warrior	1	3	26	62372	2670	381	\N	1	dps	1
1312	SA	SA2	Horde	Shaman	0	5	44	16222	95175	441	1	\N	heal	1
1313	SA	SA2	Alliance	Monk	0	3	29	2543	106000	388	\N	1	heal	1
1314	SA	SA2	Alliance	Warrior	0	4	28	45501	4113	382	\N	1	dps	1
1315	SA	SA2	Horde	Warrior	4	2	47	71075	10608	442	1	\N	dps	1
1316	SA	SA2	Alliance	Death Knight	0	2	17	51236	82487	347	\N	1	dps	1
1317	SA	SA2	Alliance	Paladin	6	3	32	118000	11454	399	\N	1	dps	1
1318	SA	SA2	Alliance	Mage	0	5	31	40404	7692	388	\N	1	dps	1
1319	SA	SA2	Horde	Druid	1	1	39	32401	1799	442	1	\N	dps	1
1320	SA	SA2	Alliance	Hunter	4	6	26	47278	11343	382	\N	1	dps	1
1321	SA	SA2	Horde	Druid	0	1	45	4553	126000	444	1	\N	heal	1
1322	SA	SA2	Horde	Demon Hunter	5	3	44	86857	9856	661	1	\N	dps	1
1323	SA	SA2	Horde	Warlock	8	2	49	117000	33117	676	1	\N	dps	1
1324	SA	SA2	Horde	Warrior	5	3	45	42009	2902	443	1	\N	dps	1
1325	SA	SA2	Alliance	Warrior	6	4	32	150000	44955	391	\N	1	dps	1
1326	SA	SA2	Alliance	Warlock	5	1	29	89119	21084	382	\N	1	dps	1
1327	SA	SA2	Horde	Warlock	3	4	37	29734	20677	423	1	\N	dps	1
1328	SA	SA2	Horde	Warrior	8	4	45	107000	18273	460	1	\N	dps	1
1329	SA	SA2	Horde	Warlock	5	0	47	77736	22259	667	1	\N	dps	1
1330	BG	BG19	Alliance	Druid	0	2	0	10074	4816	157	\N	1	dps	\N
1331	BG	BG19	Alliance	Warrior	4	5	15	71185	11257	288	\N	1	dps	\N
1332	BG	BG19	Horde	Death Knight	2	2	19	38517	12019	349	1	\N	dps	\N
1333	BG	BG19	Alliance	Shaman	2	9	11	44722	15998	275	\N	1	dps	\N
1334	BG	BG19	Horde	Priest	8	3	42	73371	23049	545	1	\N	dps	\N
1335	BG	BG19	Horde	Shaman	3	4	37	49117	12173	535	1	\N	dps	\N
1336	BG	BG19	Alliance	Paladin	4	5	16	75507	26308	291	\N	1	dps	\N
1337	BG	BG19	Horde	Shaman	1	0	41	11125	93108	543	1	\N	heal	\N
1338	BG	BG19	Horde	Priest	0	1	39	34919	180000	539	1	\N	heal	\N
1339	BG	BG19	Horde	Mage	3	2	44	51810	9167	399	1	\N	dps	\N
1340	BG	BG19	Alliance	Hunter	0	3	17	59493	7407	293	\N	1	dps	\N
1341	BG	BG19	Alliance	Hunter	1	4	17	53529	9402	293	\N	1	dps	\N
1342	BG	BG19	Alliance	Paladin	2	3	18	10837	99339	298	\N	1	heal	\N
1343	BG	BG19	Horde	Mage	5	1	43	55292	14003	547	1	\N	dps	\N
1344	BG	BG19	Horde	Shaman	1	3	39	47545	20358	389	1	\N	dps	\N
1345	BG	BG19	Horde	Rogue	12	1	44	74339	11455	549	1	\N	dps	\N
1346	BG	BG19	Horde	Priest	7	1	37	63264	14994	535	1	\N	dps	\N
1347	BG	BG19	Alliance	Druid	2	3	18	65380	8580	299	\N	1	dps	\N
1348	BG	BG19	Alliance	Rogue	2	9	13	57711	10011	277	\N	1	dps	\N
1349	SM	SM7	Alliance	Warrior	7	0	28	62049	5084	376	\N	1	dps	\N
1350	SM	SM7	Horde	Priest	0	4	25	20541	34936	359	1	\N	heal	\N
1351	SM	SM7	Alliance	Priest	0	1	30	26040	139000	384	\N	1	heal	\N
1352	SM	SM7	Alliance	Hunter	1	7	18	35973	13605	349	\N	1	dps	\N
1353	SM	SM7	Horde	Shaman	0	5	25	8524	50325	509	1	\N	heal	\N
1354	SM	SM7	Alliance	Rogue	7	2	27	56005	6760	375	\N	1	dps	\N
1355	SM	SM7	Alliance	Mage	2	3	26	77019	20339	369	\N	1	dps	\N
1356	SM	SM7	Horde	Rogue	3	6	22	53265	4637	353	1	\N	dps	\N
1357	SM	SM7	Horde	Demon Hunter	5	2	24	51010	2176	357	1	\N	dps	\N
1358	SM	SM7	Alliance	Death Knight	1	5	19	28508	71025	351	\N	1	dps	\N
1359	SM	SM7	Horde	Druid	2	3	28	62414	3183	515	1	\N	dps	\N
1360	SM	SM7	Alliance	Mage	1	1	25	34504	16971	369	\N	1	dps	\N
1361	SM	SM7	Horde	Monk	0	1	28	6564	113000	365	1	\N	heal	\N
1362	SM	SM7	Alliance	Priest	2	2	27	4183	79406	373	\N	1	heal	\N
1363	SM	SM7	Horde	Druid	3	3	26	71599	10320	511	1	\N	dps	\N
1364	SM	SM7	Alliance	Warrior	3	6	23	38107	3818	364	\N	1	dps	\N
1365	SM	SM7	Horde	Rogue	1	2	22	27340	2257	503	1	\N	dps	\N
1366	SM	SM7	Alliance	Rogue	6	1	28	58897	7859	376	\N	1	dps	\N
1367	SM	SM7	Horde	Rogue	0	3	20	28182	2507	349	1	\N	dps	\N
1368	SM	SM7	Horde	Priest	6	1	28	161000	29382	515	1	\N	dps	\N
1369	SM	SM8	Alliance	Monk	0	6	2	3006	59675	161	\N	1	heal	\N
1370	SM	SM8	Horde	Hunter	2	1	33	18152	3181	364	1	\N	dps	\N
1371	SM	SM8	Horde	Priest	0	1	33	4486	70555	364	1	\N	heal	\N
1372	SM	SM8	Horde	Shaman	2	0	38	15017	78285	374	1	\N	heal	\N
1373	SM	SM8	Horde	Rogue	9	0	38	44634	829	374	1	\N	dps	\N
1374	SM	SM8	Alliance	Hunter	0	5	3	37600	6083	164	\N	1	dps	\N
1375	SM	SM8	Horde	Paladin	4	0	38	54113	17632	524	1	\N	dps	\N
1376	SM	SM8	Alliance	Paladin	1	7	2	2226	2014	162	\N	1	dps	\N
1377	SM	SM8	Alliance	Hunter	1	7	2	42630	8940	162	\N	1	dps	\N
1378	SM	SM8	Alliance	Hunter	0	5	2	53756	8635	161	\N	1	dps	\N
1379	SM	SM8	Alliance	Mage	0	6	2	29502	15345	161	\N	1	dps	\N
1380	SM	SM8	Horde	Warlock	4	0	36	39326	14253	370	1	\N	dps	\N
1381	SM	SM8	Alliance	Paladin	0	2	3	4929	21014	164	\N	1	heal	\N
1382	SM	SM8	Horde	Mage	4	1	39	52249	8475	378	1	\N	dps	\N
1383	SM	SM8	Horde	Demon Hunter	8	0	36	79997	14178	520	1	\N	dps	\N
1384	SM	SM8	Alliance	Hunter	1	0	3	19343	6967	164	\N	1	dps	\N
1385	SM	SM8	Horde	Shaman	4	0	35	38874	9407	370	1	\N	dps	\N
1386	SM	SM8	Alliance	Demon Hunter	1	0	3	16103	20932	164	\N	1	dps	\N
1387	SM	SM8	Horde	Paladin	0	0	32	10689	6333	362	1	\N	dps	\N
1388	SM	SM8	Alliance	Mage	0	1	3	4043	0	164	\N	1	dps	\N
1389	TP	TP6	Horde	Paladin	11	3	38	60589	29728	447	1	\N	dps	\N
1390	TP	TP6	Alliance	Hunter	3	7	22	37892	14136	285	\N	1	dps	\N
1391	TP	TP6	Horde	Paladin	3	3	29	52299	20458	426	1	\N	dps	\N
1392	TP	TP6	Horde	Priest	5	3	24	25578	97152	408	1	\N	heal	\N
1393	TP	TP6	Horde	Mage	3	6	28	41167	16543	423	1	\N	dps	\N
1394	TP	TP6	Horde	Priest	4	2	33	69036	7661	432	1	\N	dps	\N
1395	TP	TP6	Horde	Shaman	0	3	27	13551	83287	416	1	\N	heal	\N
1396	TP	TP6	Horde	Druid	6	3	35	54651	14063	441	1	\N	dps	\N
1397	TP	TP6	Horde	Rogue	4	4	26	47028	1792	416	1	\N	dps	\N
1398	TP	TP6	Alliance	Death Knight	7	3	31	88260	26169	327	\N	1	dps	\N
1399	TP	TP6	Alliance	Demon Hunter	8	3	26	64300	6495	296	\N	1	dps	\N
1400	TP	TP6	Alliance	Demon Hunter	2	6	24	67914	7705	292	\N	1	dps	\N
1401	TP	TP6	Alliance	Demon Hunter	2	7	22	73619	6761	282	\N	1	dps	\N
1402	TP	TP6	Horde	Druid	3	1	28	28247	5040	418	1	\N	dps	\N
1403	TP	TP6	Alliance	Shaman	0	3	25	6022	64603	292	\N	1	heal	\N
1404	TP	TP6	Horde	Warrior	8	8	27	88575	11014	422	1	\N	dps	\N
1405	TP	TP6	Alliance	Druid	3	4	16	26248	0	261	\N	1	dps	\N
1406	TP	TP6	Alliance	Rogue	7	3	30	65592	10581	315	\N	1	dps	\N
1407	TP	TP6	Alliance	Hunter	2	7	26	32478	6614	299	\N	1	dps	\N
1408	TP	TP6	Alliance	Monk	0	4	19	671	68021	275	\N	1	heal	\N
1409	WG	WG18	Alliance	Druid	1	1	32	75407	3350	271	\N	1	dps	\N
1410	WG	WG18	Alliance	Demon Hunter	12	1	41	122000	9439	289	\N	1	dps	\N
1412	WG	WG18	Alliance	Hunter	5	1	40	78722	0	295	\N	1	dps	\N
1413	WG	WG18	Alliance	Shaman	1	0	40	17302	145000	295	\N	1	heal	\N
1414	WG	WG18	Alliance	Rogue	4	0	40	45394	7143	287	\N	1	dps	\N
1415	WG	WG18	Horde	Shaman	0	5	9	8642	59087	358	1	\N	heal	\N
1416	WG	WG18	Alliance	Warlock	0	4	28	15033	14354	264	\N	1	dps	\N
1417	WG	WG18	Alliance	Demon Hunter	5	1	35	87797	4193	274	\N	1	dps	\N
1418	WG	WG18	Horde	Warrior	3	5	9	78488	12640	508	1	\N	dps	\N
1419	WG	WG18	Horde	Demon Hunter	3	4	10	46920	10159	518	1	\N	dps	\N
1420	WG	WG18	Alliance	Hunter	4	2	29	56994	7839	265	\N	1	dps	\N
1421	WG	WG18	Horde	Shaman	0	2	8	70448	12826	504	1	\N	dps	\N
1422	WG	WG18	Horde	Priest	1	6	8	6956	113000	355	1	\N	heal	\N
1423	WG	WG18	Horde	Priest	3	3	10	30447	19588	510	1	\N	dps	\N
1424	WG	WG18	Horde	Warrior	3	5	8	71391	7063	355	1	\N	dps	\N
1425	WG	WG18	Horde	Shaman	1	6	8	39354	18043	506	1	\N	dps	\N
1426	WG	WG18	Horde	Shaman	0	2	8	2104	88411	506	1	\N	heal	\N
1427	WG	WG18	Alliance	Monk	0	0	38	842	54608	287	\N	1	heal	\N
1428	WG	WG18	Alliance	Paladin	6	3	26	68877	19716	260	\N	1	dps	\N
1429	TK	TK12	Alliance	Hunter	0	0	3	8519	7349	153	\N	1	dps	\N
1430	TK	TK12	Horde	Demon Hunter	1	0	26	23318	1301	503	1	\N	dps	\N
1431	TK	TK12	Alliance	Druid	0	2	3	12462	7766	153	\N	1	dps	\N
1432	TK	TK12	Alliance	Hunter	0	3	1	59714	4953	145	\N	1	dps	\N
1433	TK	TK12	Horde	Priest	1	0	26	6914	98271	353	1	\N	heal	\N
1434	TK	TK12	Horde	Death Knight	5	0	26	39247	8481	503	1	\N	dps	\N
1435	TK	TK12	Horde	Shaman	0	1	24	2350	35636	499	1	\N	heal	\N
1436	TK	TK12	Horde	Warlock	1	0	26	16071	1684	353	1	\N	dps	\N
1437	TK	TK12	Alliance	Rogue	0	4	2	36982	3574	149	\N	1	dps	\N
1438	TK	TK12	Horde	Paladin	4	0	26	27810	5621	503	1	\N	dps	\N
1439	TK	TK12	Alliance	Warrior	0	3	1	11540	1653	146	\N	1	dps	\N
1440	TK	TK12	Alliance	Mage	1	3	2	16299	6069	150	\N	1	dps	\N
1441	TK	TK12	Horde	Rogue	1	1	25	13529	4761	501	1	\N	dps	\N
1442	TK	TK12	Horde	Mage	7	0	26	22797	5377	501	1	\N	dps	\N
1443	TK	TK12	Alliance	Priest	0	3	2	4152	11838	149	\N	1	heal	\N
1444	TK	TK12	Horde	Warrior	3	0	26	28089	3314	501	1	\N	dps	\N
1445	TK	TK12	Alliance	Demon Hunter	1	4	3	29015	3672	153	\N	1	dps	\N
1446	TK	TK12	Alliance	Rogue	0	3	1	21828	9299	145	\N	1	dps	\N
1447	TK	TK12	Horde	Death Knight	3	1	24	14902	25871	497	1	\N	dps	\N
1448	TK	TK13	Alliance	Druid	1	5	34	579	131000	400	\N	1	heal	\N
1449	TK	TK13	Alliance	Warlock	0	7	35	28988	21865	403	\N	1	dps	\N
1450	TK	TK13	Alliance	Death Knight	7	5	38	140000	23607	410	\N	1	dps	\N
1451	TK	TK13	Horde	Demon Hunter	4	2	46	71962	19836	541	1	\N	dps	\N
1452	TK	TK13	Alliance	Hunter	6	6	37	101000	5765	408	\N	1	dps	\N
1453	TK	TK13	Horde	Monk	0	3	47	0	176000	393	1	\N	heal	\N
1454	TK	TK13	Horde	Rogue	9	2	49	78323	23932	547	1	\N	dps	\N
1455	TK	TK13	Horde	Shaman	7	5	40	99117	7920	529	1	\N	dps	\N
1456	TK	TK13	Horde	Warrior	10	4	49	89436	16007	547	1	\N	dps	\N
1457	TK	TK13	Alliance	Mage	5	8	35	94137	36356	404	\N	1	dps	\N
1458	TK	TK13	Horde	Shaman	1	5	48	10297	96458	395	1	\N	heal	\N
1459	TK	TK13	Horde	Paladin	4	4	44	44358	41300	537	1	\N	dps	\N
1460	TK	TK13	Horde	Demon Hunter	3	4	42	41994	2633	383	1	\N	dps	\N
1461	TK	TK13	Alliance	Warrior	11	4	36	68310	13563	405	\N	1	dps	\N
1462	TK	TK13	Alliance	Druid	0	1	40	1003	77531	416	\N	1	heal	\N
1463	TK	TK13	Horde	Hunter	6	7	45	82401	3816	389	1	\N	dps	\N
1464	TK	TK13	Alliance	Hunter	4	4	37	56062	6574	409	\N	1	dps	\N
1465	TK	TK13	Alliance	Hunter	2	5	33	73636	8484	400	\N	1	dps	\N
1466	TK	TK13	Alliance	Rogue	5	5	37	41053	7480	408	\N	1	dps	\N
1467	TK	TK13	Horde	Warrior	6	5	41	58584	10628	531	1	\N	dps	\N
1468	TK	TK14	Alliance	Death Knight	2	4	32	42662	8678	347	\N	1	dps	\N
1469	TK	TK14	Alliance	Mage	0	4	39	10955	2449	363	\N	1	dps	\N
1470	TK	TK14	Horde	Warlock	6	2	33	82344	24761	364	1	\N	dps	\N
1471	TK	TK14	Horde	Warrior	8	6	25	86048	9333	358	1	\N	dps	\N
1472	TK	TK14	Alliance	Monk	6	4	41	89222	18639	369	\N	1	dps	\N
1473	TK	TK14	Alliance	Mage	7	0	42	65391	8774	372	\N	1	dps	\N
1474	TK	TK14	Alliance	Paladin	4	4	38	81646	14746	360	\N	1	dps	\N
1475	TK	TK14	Alliance	Mage	6	4	35	81291	30027	354	\N	1	dps	\N
1476	TK	TK14	Horde	Shaman	0	5	33	13630	129000	374	1	\N	heal	\N
1477	TK	TK14	Alliance	Monk	0	3	40	2104	173000	368	\N	1	heal	\N
1478	TK	TK14	Alliance	Demon Hunter	11	3	42	140000	24964	373	\N	1	dps	\N
1479	TK	TK14	Horde	Druid	2	5	32	27410	25661	522	1	\N	dps	\N
1480	TK	TK14	Alliance	Demon Hunter	3	4	35	47434	6110	359	\N	1	dps	\N
1481	TK	TK14	Horde	Rogue	0	6	27	35088	3370	512	1	\N	dps	\N
1482	TK	TK14	Horde	Mage	3	5	31	61415	19862	370	1	\N	dps	\N
1483	TK	TK14	Horde	Hunter	4	5	29	44696	7467	516	1	\N	dps	\N
1484	TK	TK14	Horde	Paladin	6	2	30	130000	31887	368	1	\N	dps	\N
1485	TK	TK14	Alliance	Shaman	1	3	40	3787	81047	366	\N	1	heal	\N
1486	TK	TK14	Horde	Mage	4	3	30	76612	10672	368	1	\N	dps	\N
1487	TK	TK14	Horde	Druid	0	4	30	147	110000	518	1	\N	heal	\N
1488	TK	TK15	Alliance	Druid	0	2	17	3014	80452	244	\N	1	heal	\N
1489	TK	TK15	Horde	Druid	8	1	32	107000	1044	517	1	\N	dps	\N
1490	TK	TK15	Alliance	Warlock	2	5	13	43744	18637	232	\N	1	dps	\N
1491	TK	TK15	Horde	Warrior	7	1	30	53753	11686	513	1	\N	dps	\N
1492	TK	TK15	Horde	Demon Hunter	2	3	30	43392	11974	513	1	\N	dps	\N
1493	TK	TK15	Alliance	Hunter	4	5	17	64628	6708	244	\N	1	dps	\N
1494	TK	TK15	Alliance	Shaman	0	3	14	8494	82772	237	\N	1	heal	\N
1495	TK	TK15	Alliance	Demon Hunter	2	3	17	62322	5202	244	\N	1	dps	\N
1496	TK	TK15	Horde	Shaman	1	3	25	11395	84427	503	1	\N	heal	\N
1497	TK	TK15	Horde	Priest	0	2	32	0	160000	517	1	\N	heal	\N
1498	TK	TK15	Alliance	Priest	0	4	19	18455	111000	251	\N	1	heal	\N
1499	TK	TK15	Horde	Mage	2	1	30	52389	3678	513	1	\N	dps	\N
1500	TK	TK15	Horde	Warlock	4	1	33	80137	28552	519	1	\N	dps	\N
1501	TK	TK15	Alliance	Shaman	5	3	17	95019	14874	244	\N	1	dps	\N
1502	TK	TK15	Horde	Warrior	2	4	24	43722	1994	501	1	\N	dps	\N
1503	TK	TK15	Alliance	Paladin	1	3	18	68113	9642	244	\N	1	dps	\N
1504	TK	TK15	Horde	Death Knight	5	2	33	75589	31472	519	1	\N	dps	\N
1505	TK	TK15	Horde	Mage	1	2	31	52493	7418	365	1	\N	dps	\N
1506	TK	TK15	Alliance	Demon Hunter	2	1	18	46257	18557	248	\N	1	dps	\N
1507	TK	TK15	Alliance	Rogue	4	3	19	42048	15587	250	\N	1	dps	\N
1508	TK	TK16	Alliance	Paladin	1	1	9	530	24974	175	\N	1	heal	\N
1509	TK	TK16	Alliance	Paladin	0	1	10	12796	5987	192	\N	1	dps	\N
1510	TK	TK16	Horde	Paladin	0	3	48	2934	157000	402	1	\N	heal	\N
1511	TK	TK16	Horde	Warlock	10	3	46	83174	23858	548	1	\N	dps	\N
1512	TK	TK16	Horde	Warrior	9	5	39	49415	5452	384	1	\N	dps	\N
1513	TK	TK16	Horde	Druid	2	4	46	30242	4069	548	1	\N	dps	\N
1514	TK	TK16	Alliance	Rogue	4	1	14	36795	2088	219	\N	1	dps	\N
1515	TK	TK16	Horde	Warrior	10	4	47	48192	14157	550	1	\N	dps	\N
1516	TK	TK16	Alliance	Paladin	4	7	30	65098	26829	322	\N	1	dps	\N
1517	TK	TK16	Horde	Druid	5	3	45	40686	20344	546	1	\N	dps	\N
1518	TK	TK16	Alliance	Hunter	5	4	27	125000	12438	311	\N	1	dps	\N
1519	TK	TK16	Horde	Shaman	0	4	45	17235	130000	396	1	\N	heal	\N
1520	TK	TK16	Alliance	Druid	2	6	26	64656	12513	309	\N	1	dps	\N
1521	TK	TK16	Alliance	Hunter	2	7	24	36981	10817	305	\N	1	dps	\N
1522	TK	TK16	Horde	Warlock	8	2	45	73862	15394	396	1	\N	dps	\N
1523	TK	TK16	Horde	Monk	1	2	27	9347	2770	473	1	\N	dps	\N
1524	TK	TK16	Alliance	Shaman	0	4	23	3754	24783	301	\N	1	heal	\N
1525	TK	TK16	Alliance	Druid	7	7	27	63084	7388	314	\N	1	dps	\N
1526	TK	TK16	Horde	Mage	5	0	51	42275	10795	558	1	\N	dps	\N
1527	BG	BG20	Alliance	Warlock	9	3	18	92912	22203	765	1	\N	dps	\N
1528	BG	BG20	Horde	Warrior	8	2	26	54009	6304	231	\N	1	dps	\N
1529	BG	BG20	Alliance	Druid	1	1	14	1464	149000	528	1	\N	heal	\N
1530	BG	BG20	Alliance	Demon Hunter	1	3	15	85182	13961	230	1	\N	dps	\N
1531	BG	BG20	Horde	Warlock	3	2	29	81150	49207	240	\N	1	dps	\N
1532	BG	BG20	Alliance	Monk	0	1	16	6874	66161	538	1	\N	heal	\N
1533	BG	BG20	Horde	Rogue	2	1	24	95626	2923	229	\N	1	dps	\N
1534	BG	BG20	Horde	Druid	3	4	22	65587	10163	223	\N	1	dps	\N
1535	BG	BG20	Alliance	Mage	3	5	13	55314	14836	522	1	\N	dps	\N
1536	BG	BG20	Alliance	Demon Hunter	3	4	15	68559	41916	757	1	\N	dps	\N
1537	BG	BG20	Horde	Shaman	0	2	25	13709	107000	230	\N	1	heal	\N
1538	BG	BG20	Horde	Death Knight	0	3	17	26724	23131	213	\N	1	dps	\N
1539	BG	BG20	Alliance	Priest	1	3	4	13538	3520	720	1	\N	dps	\N
1540	BG	BG20	Alliance	Priest	0	2	16	3286	75885	533	1	\N	heal	\N
1541	BG	BG20	Horde	Druid	2	1	21	38719	21763	222	\N	1	dps	\N
1542	BG	BG20	Alliance	Monk	1	2	15	37484	20293	759	1	\N	dps	\N
1543	BG	BG20	Alliance	Mage	0	5	13	27179	12156	523	1	\N	dps	\N
1544	BG	BG20	Horde	Death Knight	5	3	15	47421	20308	208	\N	1	dps	\N
1545	BG	BG20	Horde	Death Knight	8	2	26	117000	16546	231	\N	1	dps	\N
1546	BG	BG20	Horde	Rogue	0	2	23	34046	23907	227	\N	1	dps	\N
1547	WG	WG19	Alliance	Shaman	1	8	18	23832	3299	251	\N	1	dps	\N
1548	WG	WG19	Horde	Warrior	10	2	58	49147	4812	435	1	\N	dps	\N
1549	WG	WG19	Horde	Death Knight	8	2	59	72011	31079	437	1	\N	dps	\N
1550	WG	WG19	Alliance	Warrior	2	7	16	51305	9969	243	\N	1	dps	\N
1551	WG	WG19	Horde	Demon Hunter	6	4	61	99829	11649	441	1	\N	dps	\N
1552	WG	WG19	Alliance	Warrior	0	0	1	4518	286	131	\N	1	dps	\N
1553	WG	WG19	Alliance	Warrior	3	8	20	68925	16263	257	\N	1	dps	\N
1554	WG	WG19	Horde	Paladin	12	2	61	103000	38653	591	1	\N	dps	\N
1555	WG	WG19	Horde	Demon Hunter	1	3	55	64733	8266	429	1	\N	dps	\N
1556	WG	WG19	Horde	Shaman	1	2	59	13751	116000	437	1	\N	heal	\N
1557	WG	WG19	Alliance	Monk	0	6	17	1562	111000	245	\N	1	heal	\N
1558	WG	WG19	Alliance	Mage	0	8	13	42564	31270	240	\N	1	dps	\N
1559	WG	WG19	Alliance	Druid	3	6	16	67213	18236	243	\N	1	dps	\N
1560	WG	WG19	Horde	Death Knight	6	2	59	71140	11555	437	1	\N	dps	\N
1561	WG	WG19	Alliance	Mage	7	3	21	54032	19697	261	\N	1	dps	\N
1562	WG	WG19	Horde	Hunter	11	3	55	87636	8537	429	1	\N	dps	\N
1563	WG	WG19	Alliance	Warrior	1	5	14	57523	16802	222	\N	1	dps	\N
1564	WG	WG19	Alliance	Warlock	3	9	18	68136	36597	252	\N	1	dps	\N
1565	WG	WG19	Horde	Hunter	2	1	59	55810	3470	437	1	\N	dps	\N
1566	WG	WG19	Horde	Paladin	0	0	63	10094	90734	445	1	\N	heal	\N
1567	WG	WG20	Alliance	Druid	2	4	17	40406	7704	300	1	\N	dps	\N
1568	WG	WG20	Horde	Warlock	4	1	31	41933	11671	190	\N	1	dps	\N
1569	WG	WG20	Alliance	Hunter	3	2	23	70607	17802	550	1	\N	dps	\N
1570	WG	WG20	Alliance	Priest	0	4	18	1875	72206	528	1	\N	heal	\N
1571	WG	WG20	Alliance	Priest	1	2	22	18626	132000	539	1	\N	heal	\N
1572	WG	WG20	Horde	Warlock	9	2	42	101000	26322	230	\N	1	dps	\N
1573	WG	WG20	Horde	Shaman	0	2	36	5964	85374	210	\N	1	heal	\N
1574	WG	WG20	Horde	Hunter	7	4	41	87912	660	228	\N	1	dps	\N
1575	WG	WG20	Horde	Death Knight	6	1	36	81925	25180	210	\N	1	dps	\N
1576	WG	WG20	Horde	Warrior	11	5	39	78293	21037	222	\N	1	dps	\N
1577	WG	WG20	Horde	Shaman	0	1	42	28514	117000	231	\N	1	heal	\N
1578	WG	WG20	Alliance	Hunter	1	6	19	79143	11405	533	1	\N	dps	\N
1579	WG	WG20	Alliance	Rogue	1	5	21	42030	10436	541	1	\N	dps	\N
1580	WG	WG20	Horde	Mage	4	3	32	59822	18359	124	\N	1	dps	\N
1581	WG	WG20	Alliance	Warrior	7	7	19	32447	8385	530	1	\N	dps	\N
1582	WG	WG20	Horde	Rogue	4	3	39	28167	10817	226	\N	1	dps	\N
1583	WG	WG20	Alliance	Warrior	2	5	17	41131	8204	530	1	\N	dps	\N
1584	WG	WG20	Horde	Hunter	4	3	31	59075	4520	200	\N	1	dps	\N
1585	WG	WG20	Alliance	Paladin	5	6	21	77817	30302	541	1	\N	dps	\N
1586	WG	WG20	Alliance	Rogue	3	9	17	61062	10867	522	1	\N	dps	\N
1587	ES	ES4	Horde	Priest	2	0	56	35540	67347	549	1	\N	heal	\N
1588	ES	ES4	Alliance	Demon Hunter	5	1	13	27259	4234	249	\N	1	dps	\N
1589	ES	ES4	Alliance	Hunter	2	0	15	48582	5392	296	\N	1	dps	\N
1590	ES	ES4	Horde	Priest	1	0	56	1674	106000	542	1	\N	heal	\N
1591	ES	ES4	Horde	Warrior	1	2	36	9826	4626	534	1	\N	dps	\N
1592	ES	ES4	Alliance	Rogue	2	4	8	27857	9137	288	\N	1	dps	\N
1593	ES	ES4	Horde	Druid	9	2	53	85242	1848	558	1	\N	dps	\N
1594	ES	ES4	Alliance	Monk	0	5	12	17784	59506	291	\N	1	heal	\N
1595	ES	ES4	Horde	Demon Hunter	1	1	4	8937	2359	374	1	\N	dps	\N
1596	ES	ES4	Alliance	Priest	1	0	17	25951	141000	301	\N	1	heal	\N
1597	ES	ES4	Alliance	Rogue	3	4	11	46547	5329	297	\N	1	dps	\N
1598	ES	ES4	Alliance	Warrior	2	9	10	56473	2510	298	\N	1	dps	\N
1599	ES	ES4	Horde	Shaman	5	3	44	45863	21908	557	1	\N	dps	\N
1600	ES	ES4	Horde	Shaman	3	1	54	10794	46407	548	1	\N	heal	\N
1601	ES	ES4	Alliance	Paladin	1	2	16	17147	13681	313	\N	1	dps	\N
1602	ES	ES4	Alliance	Paladin	0	8	12	26178	16534	292	\N	1	dps	\N
1603	ES	ES4	Alliance	Paladin	0	5	15	30744	10957	300	\N	1	dps	\N
1604	ES	ES4	Horde	Rogue	0	1	39	25085	3507	524	1	\N	dps	\N
1605	ES	ES4	Horde	Death Knight	5	2	47	54951	10846	537	1	\N	dps	\N
1606	ES	ES4	Alliance	Druid	0	3	8	2490	44501	285	\N	1	heal	\N
1607	ES	ES4	Horde	Mage	5	1	55	70075	8733	691	1	\N	dps	\N
1608	ES	ES4	Horde	Hunter	0	5	28	7213	2449	525	1	\N	dps	\N
1609	ES	ES4	Horde	Hunter	3	1	46	39941	5455	535	1	\N	dps	\N
1610	ES	ES4	Horde	Druid	5	0	53	54495	9757	544	1	\N	dps	\N
1611	ES	ES4	Alliance	Druid	0	2	12	0	68020	288	\N	1	heal	\N
1612	ES	ES4	Horde	Warrior	14	0	53	91140	8135	688	1	\N	dps	\N
1613	ES	ES4	Alliance	Demon Hunter	1	5	15	29305	7317	302	\N	1	dps	\N
1614	ES	ES4	Horde	Mage	3	2	43	53892	6475	686	1	\N	dps	\N
1615	ES	ES4	Alliance	Demon Hunter	1	4	9	36178	4972	281	\N	1	dps	\N
1616	ES	ES4	Alliance	Warrior	2	9	8	28963	7506	290	\N	1	dps	\N
1617	SM	SM9	Alliance	Hunter	2	5	7	46895	11397	205	\N	1	dps	\N
1618	SM	SM9	Horde	Hunter	2	1	42	38863	9133	378	1	\N	dps	\N
1619	SM	SM9	Alliance	Demon Hunter	0	6	5	25538	6860	198	\N	1	dps	\N
1620	SM	SM9	Alliance	Druid	0	3	8	1959	28827	207	\N	1	heal	\N
1621	SM	SM9	Horde	Paladin	5	1	44	48458	12896	381	1	\N	dps	\N
1622	SM	SM9	Horde	Monk	4	1	34	56851	18138	512	1	\N	dps	\N
1623	SM	SM9	Horde	Shaman	0	2	40	12240	79175	523	1	\N	heal	\N
1624	SM	SM9	Alliance	Warlock	1	4	10	50085	16963	212	\N	1	dps	\N
1625	SM	SM9	Horde	Shaman	9	0	44	58348	2722	531	1	\N	dps	\N
1626	SM	SM9	Horde	Rogue	3	2	41	31422	8496	525	1	\N	dps	\N
1627	SM	SM9	Horde	Paladin	10	0	43	54106	7856	379	1	\N	dps	\N
1628	SM	SM9	Alliance	Druid	0	0	2	5307	2574	161	\N	1	dps	\N
1629	SM	SM9	Alliance	Rogue	1	3	10	27988	4721	216	\N	1	dps	\N
1630	SM	SM9	Alliance	Paladin	2	6	6	58911	13210	204	\N	1	dps	\N
1631	SM	SM9	Alliance	Monk	0	8	9	1892	49708	211	\N	1	heal	\N
1632	SM	SM9	Horde	Warlock	7	1	42	46477	24408	527	1	\N	dps	\N
1633	SM	SM9	Horde	Priest	1	1	45	15925	106000	534	1	\N	heal	\N
1634	SM	SM9	Alliance	Mage	3	5	7	56131	27128	205	\N	1	dps	\N
1635	SM	SM9	Horde	Demon Hunter	3	1	44	36959	8122	532	1	\N	dps	\N
1636	SM	SM9	Alliance	Hunter	1	2	10	18723	151	216	\N	1	dps	\N
1637	SM	SM10	Horde	Druid	2	2	43	19709	7992	381	1	\N	dps	\N
1638	SM	SM10	Horde	Warlock	12	0	51	92666	17003	547	1	\N	dps	\N
1639	SM	SM10	Alliance	Demon Hunter	0	4	14	43245	3612	284	\N	1	dps	\N
1640	SM	SM10	Horde	Priest	10	1	49	83088	14028	542	1	\N	dps	\N
1641	SM	SM10	Horde	Priest	0	0	51	2167	151000	547	1	\N	heal	\N
1642	SM	SM10	Horde	Shaman	1	3	48	12784	77834	390	1	\N	heal	\N
1643	SM	SM10	Horde	Demon Hunter	9	4	44	55059	13713	383	1	\N	dps	\N
1644	SM	SM10	Horde	Demon Hunter	3	1	47	44713	14554	539	1	\N	dps	\N
1645	SM	SM10	Alliance	Warrior	3	3	15	48150	12128	289	\N	1	dps	\N
1646	SM	SM10	Alliance	Hunter	3	7	14	65217	6619	287	\N	1	dps	\N
1647	SM	SM10	Alliance	Rogue	4	2	16	54559	7544	293	\N	1	dps	\N
1648	SM	SM10	Alliance	Hunter	0	3	15	15342	5312	289	\N	1	dps	\N
1649	SM	SM10	Alliance	Death Knight	1	7	13	74957	19339	283	\N	1	dps	\N
1650	SM	SM10	Horde	Warlock	4	1	47	45590	32278	389	1	\N	dps	\N
1651	SM	SM10	Alliance	Hunter	2	7	10	51846	10121	275	\N	1	dps	\N
1652	SM	SM10	Alliance	Rogue	2	8	10	49315	4998	273	\N	1	dps	\N
1653	SM	SM10	Alliance	Shaman	0	8	12	2218	70646	279	\N	1	heal	\N
1654	SM	SM10	Alliance	Rogue	2	3	12	27831	2080	279	\N	1	dps	\N
1655	SM	SM10	Horde	Warlock	4	4	42	33110	7823	380	1	\N	dps	\N
1656	SM	SM10	Horde	Rogue	6	1	48	43390	8537	391	1	\N	dps	\N
1657	SM	SM11	Alliance	Hunter	3	5	29	29020	8104	779	1	\N	dps	\N
1658	SM	SM11	Alliance	Paladin	0	0	43	1138	135000	582	1	\N	heal	\N
1659	SM	SM11	Horde	Druid	2	10	33	24278	11252	245	\N	1	dps	\N
1660	SM	SM11	Horde	Warrior	5	7	35	68822	22706	250	\N	1	dps	\N
1661	SM	SM11	Alliance	Death Knight	8	4	41	56494	17645	574	1	\N	dps	\N
1662	SM	SM11	Alliance	Druid	7	4	40	79975	20534	795	1	\N	dps	\N
1663	SM	SM11	Horde	Hunter	3	5	34	41539	4165	247	\N	1	dps	\N
1664	SM	SM11	Horde	Mage	4	4	33	65383	11656	215	\N	1	dps	\N
1665	SM	SM11	Alliance	Mage	5	5	41	66025	6585	798	1	\N	dps	\N
1666	SM	SM11	Horde	Hunter	11	1	40	97757	12008	259	\N	1	dps	\N
1667	SM	SM11	Alliance	Warlock	7	7	39	87053	43628	567	1	\N	dps	\N
1668	SM	SM11	Horde	Shaman	0	8	30	13526	73232	239	\N	1	heal	\N
1669	SM	SM11	Alliance	Mage	3	2	40	48697	10986	573	1	\N	dps	\N
1670	SM	SM11	Alliance	Druid	0	3	41	28395	25367	800	1	\N	dps	\N
1671	SM	SM11	Horde	Paladin	0	2	41	9906	132000	262	\N	1	heal	\N
1672	SM	SM11	Horde	Warrior	14	1	41	129000	31297	262	\N	1	dps	\N
1673	SM	SM11	Horde	Druid	0	3	38	18260	9555	255	\N	1	dps	\N
1674	SM	SM11	Alliance	Shaman	8	6	42	61452	16292	801	1	\N	dps	\N
1675	SM	SM11	Alliance	Death Knight	4	5	38	85475	21917	790	1	\N	dps	\N
1676	SM	SM11	Horde	Warlock	2	1	36	27663	14704	251	\N	1	dps	\N
1677	SM	SM12	Alliance	Warrior	4	6	10	51579	17744	232	\N	1	dps	\N
1678	SM	SM12	Horde	Monk	0	1	48	477	76318	400	1	\N	heal	\N
1679	SM	SM12	Horde	Death Knight	5	1	45	52488	12726	394	1	\N	dps	\N
1680	SM	SM12	Alliance	Rogue	0	1	8	6928	2795	225	\N	1	dps	\N
1681	SM	SM12	Horde	Warlock	6	2	35	54858	26844	374	1	\N	dps	\N
1682	SM	SM12	Horde	Shaman	3	0	47	8134	67338	547	1	\N	heal	\N
1683	SM	SM12	Horde	Warlock	5	1	44	71068	44859	541	1	\N	dps	\N
1684	SM	SM12	Alliance	Hunter	0	4	7	13473	8639	223	\N	1	dps	\N
1685	SM	SM12	Alliance	Monk	0	4	6	175	93718	218	\N	1	heal	\N
1686	SM	SM12	Horde	Mage	10	1	47	72749	12622	547	1	\N	dps	\N
1687	SM	SM12	Alliance	Demon Hunter	1	6	10	69847	8471	231	\N	1	dps	\N
1688	SM	SM12	Alliance	Mage	1	7	9	25214	690	231	\N	1	dps	\N
1689	SM	SM12	Alliance	Paladin	2	4	11	52739	13547	235	\N	1	dps	\N
1690	SM	SM12	Alliance	Death Knight	2	4	9	41278	23386	230	\N	1	dps	\N
1691	SM	SM12	Alliance	Hunter	1	9	8	39005	7851	229	\N	1	dps	\N
1692	SM	SM12	Horde	Death Knight	5	0	51	31910	17027	556	1	\N	dps	\N
1693	SM	SM12	Horde	Hunter	1	3	36	19026	3297	376	1	\N	dps	\N
1694	SM	SM12	Horde	Druid	11	1	48	72494	11068	399	1	\N	dps	\N
1695	SM	SM12	Horde	Hunter	5	2	46	42019	5480	545	1	\N	dps	\N
1696	SM	SM12	Alliance	Hunter	1	6	7	21772	8279	222	\N	1	dps	\N
1697	TP	TP7	Horde	Mage	3	3	24	25088	18318	516	1	\N	dps	\N
1698	TP	TP7	Horde	Hunter	7	4	28	82278	13106	379	1	\N	dps	\N
1699	TP	TP7	Horde	Mage	3	3	27	53144	21267	524	1	\N	dps	\N
1700	TP	TP7	Alliance	Priest	0	2	25	11515	75784	364	\N	1	heal	\N
1701	TP	TP7	Horde	Shaman	0	4	22	21656	104000	512	1	\N	heal	\N
1702	TP	TP7	Horde	Warlock	6	4	28	37288	14833	387	1	\N	dps	\N
1703	TP	TP7	Alliance	Warrior	7	5	30	51905	3915	385	\N	1	dps	\N
1704	TP	TP7	Horde	Death Knight	9	3	26	85066	23570	370	1	\N	dps	\N
1705	TP	TP7	Alliance	Paladin	5	3	22	49260	21276	352	\N	1	dps	\N
1706	TP	TP7	Alliance	Hunter	2	2	10	19523	7362	192	\N	1	dps	\N
1707	TP	TP7	Alliance	Warlock	2	5	20	86428	27863	345	\N	1	dps	\N
1708	TP	TP7	Alliance	Demon Hunter	4	7	19	65587	4955	353	\N	1	dps	\N
1709	TP	TP7	Alliance	Mage	3	1	9	40413	4569	183	\N	1	dps	\N
1710	TP	TP7	Alliance	Paladin	0	4	20	33944	25814	342	\N	1	dps	\N
1711	TP	TP7	Horde	Hunter	6	3	31	74264	8504	531	1	\N	dps	\N
1712	TP	TP7	Horde	Warlock	2	2	16	37317	11233	324	1	\N	dps	\N
1713	TP	TP7	Horde	Shaman	1	9	27	66813	16219	376	1	\N	dps	\N
1714	TP	TP7	Alliance	Demon Hunter	6	5	27	73574	17729	367	\N	1	dps	\N
1715	TP	TP7	Alliance	Paladin	1	2	25	7364	104000	356	\N	1	heal	\N
1716	TP	TP7	Horde	Priest	0	2	28	10131	106000	378	1	\N	heal	\N
1717	WG	WG21	Alliance	Warrior	4	1	10	30481	9348	230	\N	1	dps	\N
1718	WG	WG21	Alliance	Hunter	2	4	27	45931	10814	391	\N	1	dps	\N
1719	WG	WG21	Horde	Warrior	4	6	20	42194	11768	239	1	\N	dps	\N
1720	WG	WG21	Alliance	Paladin	5	4	28	75679	26515	395	\N	1	dps	\N
1721	WG	WG21	Alliance	Rogue	7	2	32	65410	5161	401	\N	1	dps	\N
1722	WG	WG21	Alliance	Hunter	3	5	30	24832	4655	396	\N	1	dps	\N
1723	WG	WG21	Horde	Mage	3	4	19	70703	7299	536	1	\N	dps	\N
1724	WG	WG21	Alliance	Demon Hunter	1	2	23	26341	32808	358	\N	1	dps	\N
1725	WG	WG21	Horde	Warrior	6	2	17	61636	10911	382	1	\N	dps	\N
1726	WG	WG21	Horde	Priest	1	4	22	49478	90330	543	1	\N	heal	\N
1727	WG	WG21	Horde	Warlock	3	3	20	85721	34723	389	1	\N	dps	\N
1728	WG	WG21	Horde	Shaman	0	6	19	12671	96060	535	1	\N	heal	\N
1729	WG	WG21	Horde	Warrior	5	2	20	60747	12318	283	1	\N	dps	\N
1730	WG	WG21	Alliance	Druid	2	1	30	46015	17244	396	\N	1	dps	\N
1731	WG	WG21	Alliance	Warlock	5	2	30	72242	31692	380	\N	1	dps	\N
1732	WG	WG21	Alliance	Warlock	3	2	28	85076	23836	374	\N	1	dps	\N
1733	WG	WG21	Horde	Demon Hunter	1	3	21	69510	7004	539	1	\N	dps	\N
1734	WG	WG21	Horde	Druid	1	3	13	36901	13615	223	1	\N	dps	\N
1735	WG	WG21	Alliance	Druid	0	2	31	7850	122000	398	\N	1	heal	\N
1736	WG	WG21	Horde	Warrior	1	1	22	43313	11677	393	1	\N	dps	\N
1737	BG	BG21	Horde	Priest	5	4	21	57338	32071	181	\N	1	dps	\N
1738	BG	BG21	Horde	Rogue	2	5	18	37104	7320	174	\N	1	dps	\N
1739	BG	BG21	Alliance	Priest	6	2	27	67061	24357	796	1	\N	dps	\N
1740	BG	BG21	Horde	Warrior	7	4	17	62960	1372	178	\N	1	dps	\N
1741	BG	BG21	Alliance	Demon Hunter	2	2	11	35149	8576	523	1	\N	dps	\N
1742	BG	BG21	Horde	Demon Hunter	1	5	14	17027	5747	163	\N	1	dps	\N
1743	BG	BG21	Horde	Warrior	5	5	19	44106	16346	174	\N	1	dps	\N
1744	BG	BG21	Alliance	Paladin	3	5	27	79415	23002	795	1	\N	dps	\N
1745	BG	BG21	Alliance	Priest	5	2	32	41252	15375	589	1	\N	dps	\N
1746	BG	BG21	Alliance	Warrior	8	3	24	58960	15858	551	1	\N	dps	\N
1747	BG	BG21	Alliance	Rogue	5	5	18	61377	7839	541	1	\N	dps	\N
1748	BG	BG21	Alliance	Druid	0	4	25	2541	64123	561	1	\N	heal	\N
1749	BG	BG21	Alliance	Death Knight	1	2	32	74840	23206	589	1	\N	dps	\N
1750	BG	BG21	Horde	Shaman	0	4	19	6917	108000	174	\N	1	heal	\N
1751	BG	BG21	Alliance	Druid	9	1	32	43627	6123	592	1	\N	dps	\N
1752	BG	BG21	Alliance	Rogue	1	0	17	26465	5247	532	1	\N	dps	\N
1753	BG	BG21	Horde	Demon Hunter	4	3	19	39029	20612	178	\N	1	dps	\N
1754	SM	SM13	Alliance	Paladin	5	6	23	46860	12074	308	\N	1	dps	\N
1755	SM	SM13	Horde	Rogue	2	2	39	16128	9101	381	1	\N	dps	\N
1756	SM	SM13	Horde	Hunter	9	2	45	80861	8210	387	1	\N	dps	\N
1757	SM	SM13	Horde	Hunter	1	1	40	17533	3826	385	1	\N	dps	\N
1758	SM	SM13	Horde	Shaman	1	5	40	8122	72029	533	1	\N	heal	\N
1759	SM	SM13	Horde	Warrior	6	4	40	57846	12311	377	1	\N	dps	\N
1760	SM	SM13	Horde	Priest	8	3	38	40097	16984	527	1	\N	dps	\N
1761	SM	SM13	Alliance	Hunter	2	5	21	24386	13311	301	\N	1	dps	\N
1762	SM	SM13	Alliance	Warlock	1	1	15	19150	7111	287	\N	1	dps	\N
1763	SM	SM13	Alliance	Warrior	4	7	21	58743	13048	307	\N	1	dps	\N
1764	SM	SM13	Alliance	Demon Hunter	3	4	25	14026	4273	317	\N	1	dps	\N
1765	SM	SM13	Alliance	Demon Hunter	5	4	20	29721	2517	271	\N	1	dps	\N
1766	SM	SM13	Horde	Demon Hunter	3	4	40	31505	10318	533	1	\N	dps	\N
1767	SM	SM13	Horde	Rogue	2	2	41	30202	6700	387	1	\N	dps	\N
1768	SM	SM13	Alliance	Priest	0	5	23	6845	87975	308	\N	1	heal	\N
1769	SM	SM13	Horde	Hunter	5	4	42	53537	7427	387	1	\N	dps	\N
1770	SM	SM13	Alliance	Death Knight	8	5	22	56507	20440	307	\N	1	dps	\N
1771	SM	SM13	Alliance	Hunter	0	6	25	30328	8811	316	\N	1	dps	\N
1772	SM	SM13	Horde	Mage	6	2	43	59004	10693	389	1	\N	dps	\N
1773	AB	AB2	Alliance	Priest	1	2	34	32083	59467	419	\N	1	heal	\N
1774	AB	AB2	Alliance	Rogue	1	0	5	3417	537	159	\N	1	dps	\N
1775	AB	AB2	Alliance	Paladin	0	5	22	10392	89132	389	\N	1	heal	\N
1776	AB	AB2	Alliance	Druid	2	0	44	29315	25538	443	\N	1	dps	\N
1777	AB	AB2	Horde	Paladin	1	2	19	19930	8052	497	1	\N	dps	\N
1778	AB	AB2	Horde	Priest	6	3	45	55998	15615	415	1	\N	dps	\N
1779	AB	AB2	Horde	Shaman	1	4	39	9539	77324	563	1	\N	heal	\N
1780	AB	AB2	Horde	Shaman	0	6	25	5416	44283	520	1	\N	heal	\N
1781	AB	AB2	Alliance	Priest	1	5	18	32872	11619	364	\N	1	dps	\N
1782	AB	AB2	Horde	Demon Hunter	3	1	31	49495	14693	382	1	\N	dps	\N
1783	AB	AB2	Horde	Druid	3	7	16	49101	15907	363	1	\N	dps	\N
1784	AB	AB2	Horde	Rogue	1	2	6	15652	3278	144	1	\N	dps	\N
1785	AB	AB2	Alliance	Shaman	2	5	25	65874	9854	401	\N	1	dps	\N
1786	AB	AB2	Horde	Paladin	3	8	22	46822	25237	505	1	\N	dps	\N
1787	AB	AB2	Alliance	Mage	5	5	33	56012	16984	409	\N	1	dps	\N
1788	AB	AB2	Alliance	Warrior	9	5	21	70882	11634	394	\N	1	dps	\N
1789	AB	AB2	Alliance	Shaman	5	4	52	52029	21458	451	\N	1	dps	\N
1790	AB	AB2	Horde	Warrior	5	8	33	55833	10002	385	1	\N	dps	\N
1791	AB	AB2	Alliance	Warrior	12	6	49	81244	15231	438	\N	1	dps	\N
1792	AB	AB2	Alliance	Mage	11	3	48	148000	11004	447	\N	1	dps	\N
1793	AB	AB2	Alliance	Monk	1	1	54	10070	144000	467	\N	1	heal	\N
1794	AB	AB2	Horde	Hunter	6	7	26	121000	17475	366	1	\N	dps	\N
1795	AB	AB2	Horde	Monk	7	4	38	57899	27259	547	1	\N	dps	\N
1796	AB	AB2	Horde	Shaman	3	8	28	73034	10883	528	1	\N	dps	\N
1797	AB	AB2	Alliance	Warrior	8	1	37	40401	8881	303	\N	1	dps	\N
1798	AB	AB2	Horde	Demon Hunter	1	4	28	59759	6750	523	1	\N	dps	\N
1799	AB	AB2	Alliance	Priest	14	5	51	94787	35131	454	\N	1	dps	\N
1800	AB	AB2	Horde	Priest	11	2	38	120000	28113	256	1	\N	dps	\N
1801	AB	AB2	Horde	Monk	0	5	33	1147	74580	388	1	\N	heal	\N
1802	AB	AB2	Alliance	Monk	3	4	52	44862	18944	452	\N	1	dps	\N
1803	AB	AB3	Alliance	Priest	6	1	26	65572	13499	256	\N	1	dps	\N
1804	AB	AB3	Horde	Warlock	2	3	12	25918	14181	490	1	\N	dps	\N
1805	AB	AB3	Alliance	Priest	4	3	28	95250	21830	256	\N	1	dps	\N
1806	AB	AB3	Alliance	Warlock	7	4	21	65438	44516	248	\N	1	dps	\N
1807	AB	AB3	Horde	Druid	0	5	16	11584	5699	504	1	\N	dps	\N
1808	AB	AB3	Alliance	Shaman	0	3	29	10295	98793	260	\N	1	heal	\N
1809	AB	AB3	Horde	Death Knight	4	5	26	90874	34388	513	1	\N	dps	\N
1810	AB	AB3	Alliance	Paladin	5	4	23	85100	16128	242	\N	1	dps	\N
1811	AB	AB3	Horde	Shaman	0	6	27	3212	87141	522	1	\N	heal	\N
1812	AB	AB3	Horde	Hunter	5	0	17	30106	3023	355	1	\N	dps	\N
1813	AB	AB3	Alliance	Demon Hunter	0	4	13	12770	181	231	\N	1	dps	\N
1814	AB	AB3	Horde	Monk	5	2	32	27577	18686	553	1	\N	dps	\N
1815	AB	AB3	Horde	Rogue	3	0	35	24657	5094	555	1	\N	dps	\N
1816	AB	AB3	Alliance	Priest	0	2	26	39760	8922	251	\N	1	dps	\N
1817	AB	AB3	Alliance	Demon Hunter	0	0	4	1442	542	151	\N	1	dps	\N
1818	AB	AB3	Horde	Mage	3	2	23	51002	14875	376	1	\N	dps	\N
1819	AB	AB3	Horde	Warrior	10	4	32	75364	18050	538	1	\N	dps	\N
1820	AB	AB3	Alliance	Druid	0	6	27	24	9599	255	\N	1	heal	\N
1821	AB	AB3	Alliance	Demon Hunter	1	5	11	48328	6010	222	\N	1	dps	\N
1822	AB	AB3	Horde	Hunter	3	1	31	31267	8617	529	1	\N	dps	\N
1823	AB	AB3	Alliance	Monk	4	2	16	41586	9695	247	\N	1	dps	\N
1824	AB	AB3	Horde	Paladin	5	3	26	58704	12034	522	1	\N	dps	\N
1825	AB	AB3	Horde	Warrior	6	3	33	45382	11934	539	1	\N	dps	\N
1826	AB	AB3	Horde	Priest	0	2	29	19896	68825	521	1	\N	heal	\N
1827	AB	AB3	Alliance	Druid	0	1	27	4401	37746	262	\N	1	heal	\N
1828	AB	AB3	Alliance	Warrior	3	3	26	41396	10980	251	\N	1	dps	\N
1829	AB	AB3	Alliance	Priest	4	3	25	22706	7557	245	\N	1	dps	\N
1830	AB	AB3	Alliance	Warrior	4	5	24	62863	8981	260	\N	1	dps	\N
1831	AB	AB3	Horde	Shaman	0	0	19	7874	60104	514	1	\N	heal	\N
1832	AB	AB3	Horde	Warlock	0	2	26	36528	22703	523	1	\N	dps	\N
1833	BG	BG22	Alliance	Hunter	1	3	5	7608	4410	156	\N	1	dps	\N
1834	BG	BG22	Horde	Hunter	0	0	16	11865	16	503	1	\N	dps	\N
1835	BG	BG22	Alliance	Warrior	1	3	5	20125	3013	158	\N	1	dps	\N
1836	BG	BG22	Alliance	Mage	3	2	7	36118	4316	164	\N	1	dps	\N
1837	BG	BG22	Horde	Paladin	3	0	19	43492	13314	512	1	\N	dps	\N
1838	BG	BG22	Horde	Warrior	6	1	12	36308	7100	494	1	\N	dps	\N
1839	BG	BG22	Horde	Priest	2	1	19	26327	56789	511	1	\N	heal	\N
1840	BG	BG22	Alliance	Demon Hunter	2	2	5	26035	42432	156	\N	1	dps	\N
1841	BG	BG22	Horde	Shaman	0	0	15	9312	22022	505	1	\N	heal	\N
1842	BG	BG22	Alliance	Warrior	0	4	5	10634	3315	156	\N	1	dps	\N
1843	BG	BG22	Horde	Druid	2	2	7	23985	529	482	1	\N	dps	\N
1844	BG	BG22	Alliance	Priest	0	3	7	6368	62483	164	\N	1	heal	\N
1845	BG	BG22	Alliance	Druid	0	2	5	12985	1507	158	\N	1	dps	\N
1846	BG	BG22	Horde	Mage	5	1	18	33923	6185	513	1	\N	dps	\N
1847	BG	BG22	Alliance	Mage	0	2	7	24119	3244	164	\N	1	dps	\N
1848	BG	BG22	Horde	Rogue	0	0	22	30312	5382	369	1	\N	dps	\N
1849	BG	BG22	Horde	Hunter	0	2	12	8750	3506	498	1	\N	dps	\N
1850	BG	BG22	Horde	Demon Hunter	4	0	15	37178	3684	509	1	\N	dps	\N
1851	BG	BG22	Alliance	Mage	0	2	7	20288	9875	164	\N	1	dps	\N
1852	BG	BG23	Horde	Priest	10	3	39	95081	19128	551	1	\N	dps	\N
1853	BG	BG23	Alliance	Druid	5	8	15	69771	15734	361	\N	1	dps	\N
1854	BG	BG23	Horde	Demon Hunter	13	2	45	89028	8402	414	1	\N	dps	\N
1855	BG	BG23	Horde	Hunter	3	2	46	83294	8986	562	1	\N	dps	\N
1856	BG	BG23	Alliance	Death Knight	2	4	9	47238	30650	319	\N	1	dps	\N
1857	BG	BG23	Horde	Druid	0	3	37	1252	120000	398	1	\N	heal	\N
1858	BG	BG23	Alliance	Warlock	4	8	15	43835	26686	369	\N	1	dps	\N
1859	BG	BG23	Horde	Shaman	0	2	32	21391	115000	389	1	\N	heal	\N
1860	BG	BG23	Alliance	Rogue	3	1	13	31195	3906	355	\N	1	dps	\N
1861	BG	BG23	Horde	Rogue	4	2	21	55218	2656	369	1	\N	dps	\N
1862	BG	BG23	Alliance	Death Knight	5	5	15	78313	29998	358	\N	1	dps	\N
1863	BG	BG23	Horde	Death Knight	4	3	45	130000	33765	410	1	\N	dps	\N
1864	BG	BG23	Alliance	Death Knight	3	6	17	49009	34908	364	\N	1	dps	\N
1865	BG	BG23	Alliance	Warrior	4	3	12	60407	9878	344	\N	1	dps	\N
1866	BG	BG23	Alliance	Shaman	1	5	20	29282	194000	377	\N	1	heal	\N
1867	BG	BG23	Horde	Rogue	5	5	23	63628	4453	522	1	\N	dps	\N
1868	BG	BG23	Alliance	Warrior	0	6	15	34361	240	361	\N	1	dps	\N
1869	BG	BG23	Alliance	Druid	0	4	15	1830	151000	361	\N	1	heal	\N
1870	BG	BG23	Horde	Hunter	9	3	39	148000	10409	404	1	\N	dps	\N
1871	BG	BG23	Horde	Hunter	4	1	45	59282	3860	410	1	\N	dps	\N
1872	SM	SM14	Horde	Rogue	1	3	17	32704	5553	495	1	\N	dps	\N
1873	SM	SM14	Alliance	Priest	0	1	28	13229	81912	320	\N	1	heal	\N
1874	SM	SM14	Horde	Paladin	1	4	17	64542	19661	495	1	\N	dps	\N
1875	SM	SM14	Alliance	Mage	0	1	28	19638	5315	320	\N	1	dps	\N
1876	SM	SM14	Alliance	Warrior	13	3	28	77108	16067	320	\N	1	dps	\N
1877	SM	SM14	Horde	Shaman	0	2	14	7782	67102	338	1	\N	heal	\N
1878	SM	SM14	Alliance	Hunter	1	2	27	36395	9152	318	\N	1	dps	\N
1879	SM	SM14	Horde	Warlock	0	1	15	6503	849	340	1	\N	dps	\N
1880	SM	SM14	Horde	Shaman	1	1	16	47946	16400	492	1	\N	dps	\N
1881	SM	SM14	Alliance	Demon Hunter	1	1	28	112000	15644	320	\N	1	dps	\N
1882	SM	SM14	Alliance	Demon Hunter	1	2	26	31128	3765	310	\N	1	dps	\N
1883	SM	SM14	Horde	Hunter	4	2	17	62828	5705	345	1	\N	dps	\N
1884	SM	SM14	Horde	Warrior	10	3	18	108000	22101	347	1	\N	dps	\N
1885	SM	SM14	Alliance	Druid	3	1	25	35865	12856	310	\N	1	dps	\N
1886	SM	SM14	Alliance	Death Knight	0	5	27	84026	16686	318	\N	1	dps	\N
1887	SM	SM14	Alliance	Priest	2	1	28	43705	103000	320	\N	1	heal	\N
1888	SM	SM14	Alliance	Monk	4	2	25	67706	25297	308	\N	1	dps	\N
1889	SM	SM14	Horde	Priest	0	3	18	25256	260000	347	1	\N	heal	\N
1890	SM	SM14	Horde	Druid	1	4	17	28922	8454	344	1	\N	dps	\N
1891	TP	TP8	Horde	Warlock	2	4	35	71878	26368	258	\N	1	dps	\N
1892	TP	TP8	Alliance	Paladin	1	3	26	954	39017	555	1	\N	heal	\N
1893	TP	TP8	Alliance	Paladin	8	2	28	100000	29124	578	1	\N	dps	\N
1894	TP	TP8	Horde	Priest	1	1	23	15828	124000	233	\N	1	heal	\N
1895	TP	TP8	Alliance	Shaman	3	7	21	24455	12843	543	1	\N	dps	\N
1896	TP	TP8	Horde	Shaman	0	6	27	18443	92832	249	\N	1	heal	\N
1897	TP	TP8	Horde	Demon Hunter	4	4	32	55542	9357	250	\N	1	dps	\N
1898	TP	TP8	Horde	Warrior	8	5	29	30900	1550	243	\N	1	dps	\N
1899	TP	TP8	Alliance	Warrior	4	8	25	51643	10127	559	1	\N	dps	\N
1900	TP	TP8	Alliance	Druid	1	3	29	6308	83198	791	1	\N	heal	\N
1901	TP	TP8	Alliance	Hunter	4	5	28	47922	10859	563	1	\N	dps	\N
1902	TP	TP8	Horde	Demon Hunter	9	2	23	74427	4347	238	\N	1	dps	\N
1903	TP	TP8	Horde	Hunter	3	7	20	37201	7869	224	\N	1	dps	\N
1904	TP	TP8	Alliance	Death Knight	3	3	31	90633	20820	813	1	\N	dps	\N
1905	TP	TP8	Alliance	Mage	5	3	22	38892	7575	754	1	\N	dps	\N
1906	TP	TP8	Horde	Paladin	6	7	27	110000	31388	249	\N	1	dps	\N
1907	TP	TP8	Horde	Hunter	8	3	31	29438	2748	246	\N	1	dps	\N
1908	TP	TP8	Horde	Demon Hunter	0	3	28	14453	2035	243	\N	1	dps	\N
1909	TP	TP8	Alliance	Rogue	1	5	30	49197	10575	569	1	\N	dps	\N
1910	TP	TP8	Alliance	Rogue	11	1	29	108000	8185	793	1	\N	dps	\N
1911	WG	WG22	Horde	Hunter	4	1	11	97295	11051	148	\N	1	dps	\N
1912	WG	WG22	Alliance	Druid	0	0	19	5346	131000	566	1	\N	heal	\N
1913	WG	WG22	Alliance	Demon Hunter	4	1	19	28905	24982	792	1	\N	dps	\N
1914	WG	WG22	Alliance	Hunter	0	5	19	21373	11905	796	1	\N	dps	\N
1915	WG	WG22	Horde	Shaman	2	1	12	5257	68760	151	\N	1	heal	\N
1916	WG	WG22	Horde	Mage	3	4	11	58701	7883	150	\N	1	dps	\N
1917	WG	WG22	Alliance	Demon Hunter	4	2	16	25613	3007	785	1	\N	dps	\N
1918	WG	WG22	Horde	Death Knight	1	0	3	3429	713	91	\N	1	dps	\N
1919	WG	WG22	Horde	Warrior	3	3	13	63508	7230	154	\N	1	dps	\N
1920	WG	WG22	Horde	Shaman	1	4	6	26308	10242	138	\N	1	dps	\N
1921	WG	WG22	Alliance	Paladin	3	1	17	59045	20714	787	1	\N	dps	\N
1922	WG	WG22	Alliance	Druid	7	0	20	76081	3545	573	1	\N	dps	\N
1923	WG	WG22	Alliance	Hunter	2	2	19	49974	5659	792	1	\N	dps	\N
1924	WG	WG22	Alliance	Priest	1	3	17	16243	43414	783	1	\N	heal	\N
1925	WG	WG22	Alliance	Paladin	5	0	22	59773	16037	803	1	\N	dps	\N
1926	WG	WG22	Horde	Death Knight	0	4	6	21801	11728	137	\N	1	dps	\N
1927	WG	WG22	Alliance	Warrior	0	1	9	15955	17154	765	1	\N	dps	\N
1928	WG	WG22	Horde	Druid	0	3	11	4546	11622	149	\N	1	heal	\N
1929	WG	WG23	Alliance	Warlock	10	3	39	131000	71560	811	1	\N	dps	\N
1930	WG	WG23	Alliance	Priest	0	3	40	24948	204000	594	1	\N	heal	\N
1931	WG	WG23	Horde	Warrior	4	5	33	69962	13217	209	\N	1	dps	\N
1932	WG	WG23	Horde	Warrior	6	5	41	84419	27094	217	\N	1	dps	\N
1933	WG	WG23	Horde	Warlock	11	5	38	157000	52790	210	\N	1	dps	\N
1934	WG	WG23	Alliance	Rogue	4	6	39	49224	12447	815	1	\N	dps	\N
1935	WG	WG23	Horde	Demon Hunter	3	7	37	49124	3666	211	\N	1	dps	\N
1936	WG	WG23	Horde	Shaman	0	6	33	13227	181000	203	\N	1	heal	\N
1937	WG	WG23	Alliance	Priest	0	1	39	11311	38052	819	1	\N	heal	\N
1938	WG	WG23	Alliance	Demon Hunter	10	3	33	115000	20511	797	1	\N	dps	\N
1939	WG	WG23	Horde	Mage	9	1	42	109000	22016	225	\N	1	dps	\N
1940	WG	WG23	Horde	Priest	1	6	35	26847	10549	204	\N	1	dps	\N
1941	WG	WG23	Alliance	Warlock	2	9	30	39385	35296	794	1	\N	dps	\N
1942	WG	WG23	Alliance	Death Knight	7	7	35	125000	36936	580	1	\N	dps	\N
1943	WG	WG23	Horde	Paladin	6	3	36	107000	25480	205	\N	1	dps	\N
1944	WG	WG23	Horde	Druid	0	7	38	40416	14201	212	\N	1	dps	\N
1945	WG	WG23	Alliance	Mage	3	5	36	123000	37924	585	1	\N	dps	\N
1946	WG	WG23	Alliance	Rogue	3	3	32	50688	19978	793	1	\N	dps	\N
1947	WG	WG23	Alliance	Rogue	8	3	36	71231	6136	809	1	\N	dps	\N
1948	WG	WG23	Horde	Priest	0	3	38	5043	159000	213	\N	1	heal	\N
1949	WG	WG24	Alliance	Warrior	8	2	29	102000	23384	779	1	\N	dps	\N
1950	WG	WG24	Alliance	Death Knight	2	3	29	82649	37455	554	1	\N	dps	\N
1951	WG	WG24	Alliance	Priest	0	4	25	4157	136000	543	1	\N	heal	\N
1952	WG	WG24	Alliance	Hunter	3	4	29	83540	11000	557	1	\N	dps	\N
1953	WG	WG24	Horde	Priest	2	3	18	116000	32554	175	\N	1	dps	\N
1954	WG	WG24	Horde	Warlock	3	5	18	88537	21718	175	\N	1	dps	\N
1955	WG	WG24	Horde	Hunter	2	4	18	47782	10529	177	\N	1	dps	\N
1956	WG	WG24	Horde	Druid	0	2	18	40418	24118	174	\N	1	dps	\N
1957	WG	WG24	Horde	Shaman	2	5	22	11027	120000	185	\N	1	heal	\N
1958	WG	WG24	Horde	Warlock	3	4	17	99698	42670	172	\N	1	dps	\N
1959	WG	WG24	Alliance	Druid	1	3	32	2247	185000	785	1	\N	heal	\N
1960	WG	WG24	Alliance	Rogue	5	3	32	64520	9753	562	1	\N	dps	\N
1961	WG	WG24	Alliance	Mage	7	1	37	99051	22809	580	1	\N	dps	\N
1962	WG	WG24	Horde	Paladin	0	1	10	8201	4193	144	\N	1	dps	\N
1963	WG	WG24	Horde	Rogue	7	4	18	53466	12543	174	\N	1	dps	\N
1964	WG	WG24	Alliance	Paladin	3	2	28	42171	15105	775	1	\N	dps	\N
1965	WG	WG24	Horde	Demon Hunter	2	4	17	21290	7684	173	\N	1	dps	\N
1966	WG	WG24	Alliance	Rogue	6	0	36	58153	7966	801	1	\N	dps	\N
1967	WG	WG24	Alliance	Mage	3	1	31	30179	13931	557	1	\N	dps	\N
1968	WG	WG24	Horde	Shaman	0	4	18	26439	107000	176	\N	1	heal	\N
1969	WG	WG25	Horde	Hunter	4	6	32	33104	13139	242	\N	1	dps	\N
1970	WG	WG25	Alliance	Warrior	7	1	24	37280	5801	646	1	\N	dps	\N
1971	WG	WG25	Alliance	Mage	6	2	52	95321	9485	855	1	\N	dps	\N
1972	WG	WG25	Alliance	Paladin	2	5	47	9978	191000	831	1	\N	heal	\N
1973	WG	WG25	Horde	Hunter	5	8	40	61001	8178	261	\N	1	dps	\N
1974	WG	WG25	Horde	Demon Hunter	3	9	35	60690	10478	249	\N	1	dps	\N
1975	WG	WG25	Alliance	Warrior	10	7	45	93892	22047	830	1	\N	dps	\N
1976	WG	WG25	Horde	Mage	4	5	35	35773	8290	249	\N	1	dps	\N
1977	WG	WG25	Horde	Rogue	2	3	19	24647	11201	177	\N	1	dps	\N
1978	WG	WG25	Horde	Shaman	0	5	37	13555	129000	253	\N	1	heal	\N
1979	WG	WG25	Alliance	Shaman	6	5	40	63042	4493	815	1	\N	dps	\N
1980	WG	WG25	Horde	Warlock	5	6	43	92077	38564	266	\N	1	dps	\N
1981	WG	WG25	Alliance	Demon Hunter	4	4	48	61627	4521	836	1	\N	dps	\N
1982	WG	WG25	Horde	Priest	11	4	41	98181	34577	262	\N	1	dps	\N
1983	WG	WG25	Alliance	Paladin	3	4	45	30481	19824	832	1	\N	dps	\N
1984	WG	WG25	Horde	Warrior	5	1	16	23913	7831	167	\N	1	dps	\N
1985	WG	WG25	Alliance	Druid	11	8	46	77764	10802	605	1	\N	dps	\N
1986	WG	WG25	Alliance	Mage	4	4	39	44593	16025	817	1	\N	dps	\N
1987	WG	WG25	Alliance	Hunter	3	6	40	37007	10112	591	1	\N	dps	\N
1988	WG	WG25	Horde	Mage	3	2	18	34636	3663	176	\N	1	dps	\N
1989	AB	AB4	Horde	Druid	0	1	37	602	89597	264	\N	1	heal	\N
1990	AB	AB4	Alliance	Hunter	0	5	18	41831	11904	539	1	\N	dps	\N
1991	AB	AB4	Alliance	Warlock	3	2	25	34687	8781	456	1	\N	dps	\N
1992	AB	AB4	Alliance	Rogue	4	1	36	46897	11402	823	1	\N	dps	\N
1993	AB	AB4	Alliance	Mage	4	2	29	43420	6721	559	1	\N	dps	\N
1994	AB	AB4	Horde	Druid	3	7	30	51786	12029	252	\N	1	dps	\N
1995	AB	AB4	Horde	Mage	3	3	36	98883	25610	262	\N	1	dps	\N
1996	AB	AB4	Alliance	Druid	6	1	15	59141	6873	534	1	\N	dps	\N
1997	AB	AB4	Alliance	Druid	2	5	28	100000	25685	543	1	\N	dps	\N
1998	AB	AB4	Horde	Shaman	0	2	2	0	5299	109	\N	1	heal	\N
1999	AB	AB4	Alliance	Monk	1	4	34	6322	80442	348	1	\N	heal	\N
2000	AB	AB4	Alliance	Demon Hunter	9	3	38	91509	13007	825	1	\N	dps	\N
2001	AB	AB4	Horde	Shaman	1	3	40	31606	138000	265	\N	1	heal	\N
2002	AB	AB4	Alliance	Monk	0	0	16	6718	157000	536	1	\N	heal	\N
2003	AB	AB4	Horde	Demon Hunter	1	5	29	26033	3158	251	\N	1	dps	\N
2004	AB	AB4	Horde	Warlock	6	4	30	94833	28101	257	\N	1	dps	\N
2005	AB	AB4	Horde	Mage	2	4	42	102000	16791	271	\N	1	dps	\N
2006	AB	AB4	Horde	Paladin	6	2	35	102000	46802	260	\N	1	dps	\N
2007	AB	AB4	Alliance	Demon Hunter	3	7	30	158000	20371	778	1	\N	dps	\N
2008	AB	AB4	Alliance	Death Knight	2	7	29	167000	34422	775	1	\N	dps	\N
2009	AB	AB4	Alliance	Paladin	1	5	36	43345	18911	579	1	\N	dps	\N
2010	AB	AB4	Horde	Warlock	4	6	25	88723	35924	240	\N	1	dps	\N
2011	AB	AB4	Alliance	Priest	1	3	36	35394	143000	579	1	\N	heal	\N
2012	AB	AB4	Alliance	Rogue	3	4	27	43327	11368	548	1	\N	dps	\N
2013	AB	AB4	Horde	Monk	0	2	46	1810	181000	271	\N	1	heal	\N
2014	AB	AB4	Horde	Monk	4	3	41	61926	35613	262	\N	1	dps	\N
2015	AB	AB4	Horde	Death Knight	5	2	37	99377	85987	267	\N	1	dps	\N
2016	AB	AB4	Horde	Warrior	11	1	39	73936	13502	270	\N	1	dps	\N
2017	AB	AB4	Alliance	Druid	7	1	41	125000	34377	599	1	\N	dps	\N
2018	AB	AB5	Alliance	Priest	2	4	13	33313	13608	357	\N	1	dps	1
2019	AB	AB5	Alliance	Priest	3	2	17	58591	21683	363	\N	1	dps	1
2020	AB	AB5	Alliance	Mage	0	1	17	502	615	373	\N	1	dps	1
2021	AB	AB5	Alliance	Druid	0	2	9	0	38909	107	\N	1	heal	1
2022	AB	AB5	Alliance	Shaman	2	5	13	39677	15426	349	\N	1	dps	1
2023	AB	AB5	Horde	Death Knight	3	0	20	18196	6609	443	1	\N	dps	1
2024	AB	AB5	Alliance	Hunter	0	2	17	35841	9123	363	\N	1	dps	1
2025	AB	AB5	Horde	Rogue	3	1	24	29986	12404	466	1	\N	dps	1
2026	AB	AB5	Horde	Warrior	8	4	27	80280	10822	462	1	\N	dps	1
2027	AB	AB5	Alliance	Hunter	1	5	11	48440	7385	344	\N	1	dps	1
2028	AB	AB5	Horde	Shaman	0	0	26	7095	93157	467	1	\N	heal	1
2029	AB	AB5	Alliance	Warlock	0	5	10	53026	20436	342	\N	1	dps	1
2030	AB	AB5	Alliance	Rogue	0	2	8	11788	5598	274	\N	1	dps	1
2031	AB	AB5	Horde	Priest	0	0	16	2048	20977	444	1	\N	heal	1
2032	AB	AB5	Horde	Priest	3	1	21	33775	8889	454	1	\N	dps	1
2033	AB	AB5	Horde	Rogue	3	1	20	25442	4900	448	1	\N	dps	1
2034	AB	AB5	Alliance	Paladin	1	1	7	33464	10869	341	\N	1	dps	1
2035	AB	AB5	Horde	Mage	5	0	25	41244	4604	248	1	\N	dps	1
2036	AB	AB5	Horde	Warlock	13	2	30	91526	67871	694	1	\N	dps	1
2037	AB	AB5	Horde	Hunter	1	1	27	34487	11551	479	1	\N	dps	1
2038	AB	AB5	Horde	Paladin	1	1	10	6076	34202	594	1	\N	heal	1
2039	AB	AB5	Horde	Paladin	2	1	23	4106	125000	685	1	\N	heal	1
2040	AB	AB5	Alliance	Warrior	0	6	8	6612	3516	347	\N	1	dps	1
2041	AB	AB5	Horde	Druid	1	2	13	18224	8181	431	1	\N	dps	1
2042	AB	AB5	Alliance	Mage	2	4	14	53676	8829	355	\N	1	dps	1
2043	AB	AB5	Alliance	Warlock	2	6	15	58611	31566	354	\N	1	dps	1
2044	AB	AB5	Alliance	Paladin	2	2	16	40503	17160	364	\N	1	dps	1
2045	AB	AB5	Alliance	Priest	4	2	15	42330	32358	359	\N	1	dps	1
2046	AB	AB5	Horde	Warlock	2	5	17	74613	46298	443	1	\N	dps	1
2047	AB	AB5	Horde	Shaman	5	2	14	27979	1694	441	1	\N	dps	1
2048	ES	ES5	Alliance	Rogue	0	0	5	25945	9115	284	\N	1	dps	1
2049	ES	ES5	Horde	Priest	4	1	68	36372	14003	619	1	\N	dps	1
2050	ES	ES5	Horde	Rogue	6	4	62	56452	11215	599	1	\N	dps	1
2051	ES	ES5	Horde	Death Knight	10	2	73	113000	42069	834	1	\N	dps	1
2052	ES	ES5	Horde	Mage	14	2	71	147000	2485	602	1	\N	dps	1
2053	ES	ES5	Horde	Demon Hunter	6	4	72	109000	24677	819	1	\N	dps	1
2054	ES	ES5	Alliance	Death Knight	0	2	4	15384	10586	259	\N	1	dps	1
2055	ES	ES5	Alliance	Warrior	5	5	20	39980	8112	377	\N	1	dps	1
2056	ES	ES5	Alliance	Rogue	1	6	18	36420	7040	378	\N	1	dps	1
2057	ES	ES5	Horde	Mage	7	2	75	93355	8303	598	1	\N	dps	1
2058	ES	ES5	Horde	Rogue	3	1	80	28011	17299	604	1	\N	dps	1
2059	ES	ES5	Alliance	Paladin	0	11	17	7286	79867	371	\N	1	heal	1
2060	ES	ES5	Horde	Shaman	0	0	67	16431	105000	843	1	\N	heal	1
2061	ES	ES5	Alliance	Paladin	2	6	16	85730	32711	364	\N	1	dps	1
2062	ES	ES5	Alliance	Paladin	1	8	17	44920	27379	370	\N	1	dps	1
2063	ES	ES5	Horde	Demon Hunter	4	2	72	65045	98260	600	1	\N	dps	1
2064	ES	ES5	Alliance	Hunter	0	2	17	11313	2808	377	\N	1	dps	1
2065	ES	ES5	Horde	Priest	2	0	80	4170	77345	610	1	\N	heal	1
2066	ES	ES5	Alliance	Hunter	1	7	17	28312	2456	367	\N	1	dps	1
2067	ES	ES5	Horde	Warrior	4	2	72	39217	9844	601	1	\N	dps	1
2068	ES	ES5	Horde	Priest	0	1	68	6288	103000	832	1	\N	heal	1
2069	ES	ES5	Horde	Mage	2	1	72	68291	14346	614	1	\N	dps	1
2070	ES	ES5	Alliance	Druid	0	5	20	8067	178000	381	\N	1	heal	1
2071	ES	ES5	Alliance	Hunter	2	3	23	68116	10678	384	\N	1	dps	1
2072	ES	ES5	Horde	Death Knight	11	1	81	46515	16633	618	1	\N	dps	1
2073	ES	ES5	Alliance	Demon Hunter	2	9	20	60160	12148	377	\N	1	dps	1
2074	ES	ES5	Horde	Hunter	9	2	72	142000	11192	603	1	\N	dps	1
2075	ES	ES5	Alliance	Mage	4	4	23	74179	12256	392	\N	1	dps	1
2076	ES	ES5	Alliance	Demon Hunter	4	6	21	93035	28102	377	\N	1	dps	1
2077	ES	ES5	Alliance	Demon Hunter	1	3	22	58282	42960	382	\N	1	dps	1
2078	SM	SM15	Alliance	Rogue	10	3	30	61190	7201	784	1	\N	dps	\N
2079	SM	SM15	Horde	Hunter	7	4	43	55385	6574	281	\N	1	dps	\N
2080	SM	SM15	Alliance	Mage	2	5	18	33515	22013	672	1	\N	dps	\N
2081	SM	SM15	Alliance	Warlock	4	5	29	87132	31513	558	1	\N	dps	\N
2082	SM	SM15	Horde	Priest	1	6	43	52068	19766	281	\N	1	dps	\N
2083	SM	SM15	Alliance	Demon Hunter	8	5	28	102000	7385	775	1	\N	dps	\N
2084	SM	SM15	Horde	Shaman	0	6	39	11253	87465	273	\N	1	heal	\N
2085	SM	SM15	Horde	Warlock	5	1	44	64520	11610	283	\N	1	dps	\N
2086	SM	SM15	Horde	Shaman	0	4	39	9266	154000	273	\N	1	heal	\N
2087	SM	SM15	Alliance	Hunter	5	4	31	57753	13648	786	1	\N	dps	\N
2088	SM	SM15	Horde	Warrior	5	3	42	41195	8195	279	\N	1	dps	\N
2089	SM	SM15	Horde	Druid	3	3	41	54202	7814	277	\N	1	dps	\N
2090	SM	SM15	Alliance	Death Knight	2	4	27	34532	13726	776	1	\N	dps	\N
2091	SM	SM15	Alliance	Druid	0	1	29	13134	3135	779	1	\N	dps	\N
2092	SM	SM15	Alliance	Druid	0	5	24	213	95407	766	1	\N	heal	\N
2093	SM	SM15	Horde	Shaman	4	4	38	66256	7771	271	\N	1	dps	\N
2094	SM	SM15	Alliance	Mage	2	7	27	85595	12037	770	1	\N	dps	\N
2095	SM	SM15	Horde	Warrior	11	1	44	100000	15169	283	\N	1	dps	\N
2096	SM	SM15	Alliance	Druid	0	3	28	752	107000	549	1	\N	heal	\N
2097	SM	SM15	Horde	Priest	8	2	44	75467	21950	283	\N	1	dps	\N
2098	TP	TP9	Alliance	Druid	0	3	48	32164	6759	613	1	\N	dps	\N
2099	TP	TP9	Alliance	Warlock	8	1	53	69468	31741	624	1	\N	dps	\N
2100	TP	TP9	Horde	Warrior	2	5	6	28747	5328	138	\N	1	dps	\N
2101	TP	TP9	Horde	Hunter	0	8	2	41108	4690	130	\N	1	dps	\N
2102	TP	TP9	Horde	Shaman	0	5	3	3355	24694	111	\N	1	heal	\N
2103	TP	TP9	Horde	Warrior	3	3	5	30147	3900	107	\N	1	dps	\N
2104	TP	TP9	Horde	Rogue	1	3	5	20680	9518	139	\N	1	dps	\N
2105	TP	TP9	Horde	Druid	0	1	2	0	6659	90	\N	1	heal	\N
2106	TP	TP9	Alliance	Mage	3	2	47	33083	4950	609	1	\N	dps	\N
2107	TP	TP9	Horde	Priest	0	9	6	12408	30377	138	\N	1	heal	\N
2108	TP	TP9	Alliance	Priest	6	1	48	67465	8015	611	1	\N	dps	\N
2109	TP	TP9	Alliance	Monk	0	0	52	654	80639	620	1	\N	heal	\N
2110	TP	TP9	Alliance	Death Knight	7	1	47	66542	12532	835	1	\N	dps	\N
2111	TP	TP9	Alliance	Druid	0	0	50	1554	66777	615	1	\N	heal	\N
2112	TP	TP9	Horde	Priest	0	2	3	7028	29418	101	\N	1	heal	\N
2113	TP	TP9	Horde	Rogue	1	1	7	26133	3317	142	\N	1	dps	\N
2114	TP	TP9	Alliance	Warlock	11	0	54	45015	18355	851	1	\N	dps	\N
2115	TP	TP9	Alliance	Rogue	8	1	51	43795	4065	617	1	\N	dps	\N
2116	TP	TP9	Alliance	Hunter	9	0	53	61122	1696	621	1	\N	dps	\N
2117	TP	TP10	Horde	Warrior	4	4	23	57327	5239	191	\N	1	dps	\N
2118	TP	TP10	Horde	Druid	0	4	14	18503	771	174	\N	1	dps	\N
2119	TP	TP10	Alliance	Rogue	4	2	34	63829	11938	637	1	\N	dps	\N
2120	TP	TP10	Alliance	Mage	5	4	28	52684	15220	843	1	\N	dps	\N
2121	TP	TP10	Horde	Hunter	3	5	24	73468	8740	193	\N	1	dps	\N
2122	TP	TP10	Horde	Shaman	0	4	22	9327	94946	190	\N	1	heal	\N
2123	TP	TP10	Horde	Demon Hunter	8	4	24	94027	6354	189	\N	1	dps	\N
2124	TP	TP10	Alliance	Warrior	3	1	27	43033	55280	845	1	\N	dps	\N
2125	TP	TP10	Alliance	Shaman	11	3	31	95820	22104	628	1	\N	dps	\N
2126	TP	TP10	Horde	Hunter	2	3	16	42075	11887	185	\N	1	dps	\N
2127	TP	TP10	Horde	Mage	0	5	23	59348	11366	192	\N	1	dps	\N
2128	TP	TP10	Horde	Priest	1	4	21	30701	101000	187	\N	1	heal	\N
2129	TP	TP10	Alliance	Warrior	7	3	32	50948	9010	854	1	\N	dps	\N
2130	TP	TP10	Alliance	Monk	4	4	32	64901	31383	651	1	\N	dps	\N
2131	TP	TP10	Alliance	Rogue	3	2	24	31966	5145	845	1	\N	dps	\N
2132	TP	TP10	Horde	Rogue	2	3	17	52094	7656	186	\N	1	dps	\N
2133	TP	TP10	Alliance	Priest	0	3	28	7001	105000	842	1	\N	heal	\N
2134	TP	TP10	Alliance	Mage	4	2	31	43500	4975	859	1	\N	dps	\N
2135	TP	TP10	Alliance	Druid	1	2	23	9793	111000	825	1	\N	heal	\N
2136	TP	TP10	Horde	Hunter	5	6	24	89454	6042	192	\N	1	dps	\N
2137	BG	BG24	Horde	Paladin	2	4	12	40804	30017	655	1	\N	dps	1
2138	BG	BG24	Horde	Rogue	11	5	25	81765	14695	696	1	\N	dps	1
2139	BG	BG24	Horde	Demon Hunter	1	7	23	52285	8208	679	1	\N	dps	1
2140	BG	BG24	Alliance	Paladin	3	5	24	61565	23349	372	\N	1	dps	1
2141	BG	BG24	Horde	Druid	2	2	32	32542	14589	696	1	\N	dps	1
2142	BG	BG24	Alliance	Warlock	3	4	19	53599	19118	341	\N	1	dps	1
2143	BG	BG24	Alliance	Demon Hunter	5	3	18	43899	13862	339	\N	1	dps	1
2144	BG	BG24	Horde	Druid	3	1	23	75242	31352	684	1	\N	dps	1
2145	BG	BG24	Horde	Rogue	8	2	30	71901	20348	467	1	\N	dps	1
2146	BG	BG24	Horde	Shaman	1	7	20	6288	83729	667	1	\N	heal	1
2147	BG	BG24	Alliance	Hunter	7	5	22	49495	13896	370	\N	1	dps	1
2148	BG	BG24	Alliance	Shaman	1	2	23	21461	22309	362	\N	1	heal	1
2149	BG	BG24	Horde	Rogue	5	4	20	36187	13063	668	1	\N	dps	1
2150	BG	BG24	Alliance	Mage	3	3	28	40131	16250	385	\N	1	dps	1
2151	BG	BG24	Horde	Shaman	8	5	25	53416	6494	678	1	\N	dps	1
2152	BG	BG24	Horde	Mage	0	5	23	25637	4594	674	1	\N	dps	1
2153	BG	BG24	Alliance	Mage	8	5	23	45830	18209	362	\N	1	dps	1
2154	BG	BG24	Alliance	Druid	5	2	11	42859	7449	299	\N	1	dps	1
2155	BG	BG24	Alliance	Mage	4	7	24	60269	15501	373	\N	1	dps	1
2156	BG	BG24	Alliance	Monk	0	5	16	1963	94972	327	\N	1	heal	1
2157	TP	TP11	Alliance	Druid	1	1	15	30292	15899	646	1	\N	dps	1
2158	TP	TP11	Alliance	Hunter	3	3	14	53421	8416	979	1	\N	dps	1
2159	TP	TP11	Alliance	Mage	3	1	16	89383	9056	648	1	\N	dps	1
2160	TP	TP11	Horde	Paladin	1	1	15	18236	9161	176	\N	1	dps	1
2161	TP	TP11	Alliance	Warrior	0	1	7	10080	9381	624	1	\N	dps	1
2162	TP	TP11	Horde	Hunter	1	2	10	16320	7573	166	\N	1	dps	1
2163	TP	TP11	Horde	Warrior	0	1	11	26229	32661	168	\N	1	dps	1
2164	TP	TP11	Alliance	Rogue	2	2	15	24402	2375	982	1	\N	dps	1
2165	TP	TP11	Alliance	Mage	2	1	16	32010	2335	985	1	\N	dps	1
2166	TP	TP11	Horde	Paladin	2	2	15	54925	13230	176	\N	1	dps	1
2167	TP	TP11	Alliance	Paladin	2	1	16	12910	56339	648	1	\N	heal	1
2168	TP	TP11	Horde	Priest	0	3	12	726	113000	170	\N	1	heal	1
2169	TP	TP11	Alliance	Hunter	2	4	15	44031	3789	982	1	\N	dps	1
2170	TP	TP11	Horde	Druid	4	0	15	67441	3785	176	\N	1	dps	1
2171	TP	TP11	Alliance	Rogue	2	0	13	37487	5877	639	1	\N	dps	1
2172	TP	TP11	Alliance	Shaman	0	1	15	6203	119000	646	1	\N	heal	1
2173	TP	TP11	Horde	Monk	0	3	11	385	27037	168	\N	1	heal	1
2174	TP	TP11	Horde	Mage	2	3	15	52602	2555	176	\N	1	dps	1
2175	TP	TP11	Horde	Paladin	3	1	15	22395	7966	176	\N	1	dps	1
2176	TP	TP12	Horde	Shaman	3	2	23	30322	5902	437	1	\N	dps	1
2177	TP	TP12	Alliance	Druid	1	1	15	61893	11907	246	\N	1	dps	1
2178	TP	TP12	Alliance	Shaman	0	2	16	2042	59842	249	\N	1	heal	1
2179	TP	TP12	Alliance	Warrior	9	2	16	44082	14123	249	\N	1	dps	1
2180	TP	TP12	Horde	Paladin	6	1	24	41294	6539	439	1	\N	dps	1
2181	TP	TP12	Alliance	Hunter	2	3	16	34196	9027	249	\N	1	dps	1
2182	TP	TP12	Horde	Shaman	0	2	22	1108	41694	435	1	\N	heal	1
2183	TP	TP12	Alliance	Hunter	2	2	16	37140	1424	249	\N	1	dps	1
2184	TP	TP12	Horde	Paladin	0	0	18	0	20974	652	1	\N	heal	1
2185	TP	TP12	Alliance	Hunter	0	1	16	6930	112	249	\N	1	dps	1
2186	TP	TP12	Horde	Warlock	0	4	23	59794	29525	437	1	\N	dps	1
2187	TP	TP12	Horde	Rogue	7	1	24	32664	3602	439	1	\N	dps	1
2188	TP	TP12	Alliance	Warrior	2	5	15	15842	3325	247	\N	1	dps	1
2189	TP	TP12	Alliance	Priest	0	5	12	1206	8114	238	\N	1	heal	1
2190	TP	TP12	Horde	Warlock	1	2	23	31111	20286	662	1	\N	dps	1
2191	TP	TP12	Horde	Death Knight	0	0	18	3065	6172	427	1	\N	dps	1
2192	TP	TP12	Alliance	Rogue	0	1	16	27145	4153	249	\N	1	dps	1
2193	TP	TP12	Horde	Warlock	6	3	24	28841	16725	664	1	\N	dps	1
2194	TP	TP12	Alliance	Hunter	0	3	11	431	0	236	\N	1	dps	1
2195	TP	TP12	Horde	Rogue	2	1	11	14925	946	413	1	\N	dps	1
2196	WG	WG26	Alliance	Warlock	0	3	8	21720	12472	179	\N	1	dps	\N
2197	WG	WG26	Alliance	Priest	0	4	6	12374	6978	175	\N	1	dps	\N
2198	WG	WG26	Horde	Death Knight	3	0	29	30050	5216	516	1	\N	dps	\N
2199	WG	WG26	Horde	Shaman	2	4	28	26034	7542	364	1	\N	dps	\N
2200	WG	WG26	Horde	Paladin	7	1	26	37652	13572	510	1	\N	dps	\N
2201	WG	WG26	Horde	Priest	3	0	29	31621	54313	366	1	\N	heal	\N
2202	WG	WG26	Horde	Priest	5	2	27	46838	7063	512	1	\N	dps	\N
2203	WG	WG26	Alliance	Rogue	0	2	7	6439	41	123	\N	1	dps	\N
2204	WG	WG26	Horde	Warrior	1	1	26	5627	658	510	1	\N	dps	\N
2205	WG	WG26	Horde	Shaman	0	1	21	4881	19278	350	1	\N	heal	\N
2206	WG	WG26	Horde	Demon Hunter	0	1	21	4595	64	500	1	\N	dps	\N
2207	WG	WG26	Alliance	Druid	1	2	7	45624	11933	175	\N	1	dps	\N
2208	WG	WG26	Alliance	Mage	6	1	10	40036	12748	185	\N	1	dps	\N
2209	WG	WG26	Alliance	Mage	1	2	9	38479	8661	181	\N	1	dps	\N
2210	WG	WG26	Alliance	Demon Hunter	0	5	8	29839	533	177	\N	1	dps	\N
2211	WG	WG26	Horde	Paladin	0	0	27	2436	51019	362	1	\N	heal	\N
2212	WG	WG26	Alliance	Mage	2	2	9	27931	6283	181	\N	1	dps	\N
2213	WG	WG26	Horde	Rogue	9	0	29	54622	12748	516	1	\N	dps	\N
2214	WG	WG26	Alliance	Paladin	0	4	6	4575	30897	171	\N	1	heal	\N
2215	WG	WG26	Alliance	Shaman	0	5	8	7150	1196	177	\N	1	dps	\N
2216	WG	WG27	Alliance	Rogue	13	0	67	67113	21108	826	1	\N	dps	1
2217	WG	WG27	Alliance	Druid	11	0	68	127000	45609	829	1	\N	dps	1
2218	WG	WG27	Alliance	Rogue	1	3	33	14963	11604	735	1	\N	dps	1
2219	WG	WG27	Alliance	Mage	7	0	66	91886	24978	823	1	\N	dps	1
2220	WG	WG27	Horde	Death Knight	1	8	10	18363	50254	206	\N	1	dps	1
2221	WG	WG27	Horde	Shaman	0	1	3	1096	26018	138	\N	1	heal	1
2222	WG	WG27	Horde	Hunter	0	9	18	42115	9199	218	\N	1	dps	1
2223	WG	WG27	Horde	Death Knight	0	9	14	35047	22679	212	\N	1	dps	1
2224	WG	WG27	Alliance	Rogue	6	5	57	41219	11188	1136	1	\N	dps	1
2225	WG	WG27	Horde	Warrior	3	7	22	35883	2770	228	\N	1	dps	1
2226	WG	WG27	Horde	Warlock	8	6	22	65190	26067	228	\N	1	dps	1
2227	WG	WG27	Alliance	Priest	11	2	63	100000	24116	1152	1	\N	dps	1
2228	WG	WG27	Alliance	Monk	3	4	49	27288	23080	771	1	\N	dps	1
2229	WG	WG27	Horde	Paladin	3	7	20	34257	21997	224	\N	1	dps	1
2230	WG	WG27	Alliance	Mage	5	3	63	60206	9793	1151	1	\N	dps	1
2231	WG	WG27	Horde	Warrior	3	4	22	39578	3890	228	\N	1	dps	1
2232	WG	WG27	Alliance	Warlock	5	1	66	52566	14074	824	1	\N	dps	1
2233	WG	WG27	Alliance	Death Knight	1	6	47	10942	4797	769	1	\N	dps	1
2234	WG	WG27	Horde	Warlock	0	9	11	23853	21136	204	\N	1	dps	1
2235	WG	WG27	Horde	Druid	1	4	10	10081	5321	206	\N	1	dps	1
2236	WG	WG28	Horde	Warrior	8	0	36	37179	5575	458	1	\N	dps	1
2237	WG	WG28	Horde	Priest	6	0	34	56082	13952	229	1	\N	dps	1
2238	WG	WG28	Horde	Hunter	6	2	33	62536	9155	677	1	\N	dps	1
2239	WG	WG28	Alliance	Hunter	0	1	5	27327	6610	242	\N	1	dps	1
2240	WG	WG28	Horde	Warlock	3	1	30	34292	8003	446	1	\N	dps	1
2241	WG	WG28	Alliance	Mage	0	3	4	17568	11550	238	\N	1	dps	1
2242	WG	WG28	Horde	Shaman	1	0	31	11763	51966	673	1	\N	heal	1
2243	WG	WG28	Alliance	Hunter	0	5	4	54160	2725	238	\N	1	dps	1
2244	WG	WG28	Alliance	Druid	0	1	2	5032	37536	203	\N	1	heal	1
2245	WG	WG28	Horde	Druid	7	0	36	87819	3206	458	1	\N	dps	1
2246	WG	WG28	Alliance	Mage	2	3	4	40439	11412	240	\N	1	dps	1
2247	WG	WG28	Horde	Monk	0	0	31	650	49358	448	1	\N	heal	1
2248	WG	WG28	Alliance	Paladin	0	5	4	32350	4453	239	\N	1	dps	1
2249	WG	WG28	Alliance	Warrior	0	5	1	11442	1206	216	\N	1	dps	1
2250	WG	WG28	Alliance	Monk	0	4	3	31	75998	236	\N	1	heal	1
2251	WG	WG28	Horde	Demon Hunter	1	1	36	63073	9017	458	1	\N	dps	1
2252	WG	WG28	Alliance	Rogue	1	2	5	29682	7451	242	\N	1	dps	1
2253	WG	WG28	Horde	Warrior	3	1	34	26300	3898	454	1	\N	dps	1
2254	WG	WG28	Horde	Priest	0	0	36	8195	115000	458	1	\N	heal	1
2255	WG	WG28	Alliance	Paladin	1	0	3	15249	18063	222	\N	1	heal	1
2256	TK	TK17	Alliance	Demon Hunter	0	3	5	30181	8397	155	\N	1	dps	\N
2257	TK	TK17	Alliance	Mage	0	1	5	35673	11267	155	\N	1	dps	\N
2258	TK	TK17	Horde	Rogue	3	0	24	34584	3903	498	1	\N	dps	\N
2259	TK	TK17	Alliance	Warlock	2	4	5	26658	9228	155	\N	1	dps	\N
2260	TK	TK17	Horde	Priest	4	0	24	66848	11947	348	1	\N	dps	\N
2261	TK	TK17	Horde	Paladin	4	1	22	41288	12433	494	1	\N	dps	\N
2262	TK	TK17	Horde	Shaman	1	1	24	4130	45949	348	1	\N	heal	\N
2263	TK	TK17	Horde	Demon Hunter	3	0	24	37983	6561	498	1	\N	dps	\N
2264	TK	TK17	Horde	Demon Hunter	1	1	20	7071	982	490	1	\N	dps	\N
2265	TK	TK17	Alliance	Mage	2	1	5	25844	10866	155	\N	1	dps	\N
2266	TK	TK17	Alliance	Monk	0	1	5	1083	54804	155	\N	1	heal	\N
2267	TK	TK17	Alliance	Warrior	1	3	4	8361	2384	152	\N	1	dps	\N
2268	TK	TK17	Alliance	Mage	0	2	4	24508	3615	152	\N	1	dps	\N
2269	TK	TK17	Horde	Rogue	2	0	24	25481	2573	498	1	\N	dps	\N
2270	TK	TK17	Alliance	Priest	0	2	3	26421	8200	149	\N	1	dps	\N
2271	TK	TK17	Horde	Priest	0	0	24	20056	83621	498	1	\N	heal	\N
2272	TK	TK17	Horde	Rogue	3	2	22	24897	1277	344	1	\N	dps	\N
2273	TK	TK17	Alliance	Druid	0	2	5	573	42840	155	\N	1	heal	\N
2274	TK	TK17	Horde	Mage	2	0	24	29245	7463	498	1	\N	dps	\N
2275	TK	TK17	Alliance	Warrior	0	5	2	5518	580	147	\N	1	dps	\N
2276	AB	AB6	Horde	Druid	6	1	59	46824	16355	573	1	\N	dps	\N
2277	AB	AB6	Alliance	Demon Hunter	3	5	27	43288	14220	257	\N	1	dps	\N
2278	AB	AB6	Alliance	Shaman	0	2	8	8459	378	157	\N	1	dps	\N
2279	AB	AB6	Alliance	Druid	2	7	19	25793	5488	252	\N	1	dps	\N
2280	AB	AB6	Horde	Paladin	5	1	61	55608	20981	438	1	\N	dps	\N
2281	AB	AB6	Horde	Warlock	13	0	47	74042	30250	566	1	\N	dps	\N
2282	AB	AB6	Alliance	Warlock	2	6	14	27485	16367	225	\N	1	dps	\N
2283	AB	AB6	Alliance	Death Knight	2	5	19	41836	18778	235	\N	1	dps	\N
2284	AB	AB6	Alliance	Hunter	0	1	2	13111	1383	131	\N	1	dps	\N
2285	AB	AB6	Alliance	Shaman	2	7	24	32683	7964	243	\N	1	dps	\N
2286	AB	AB6	Alliance	Druid	1	5	14	23175	4944	107	\N	1	dps	\N
2287	AB	AB6	Horde	Rogue	9	3	65	52803	11361	589	1	\N	dps	\N
2288	AB	AB6	Horde	Warrior	10	1	75	61062	11554	327	1	\N	dps	\N
2289	AB	AB6	Horde	Shaman	0	1	54	13191	65371	585	1	\N	heal	\N
2290	AB	AB6	Horde	Rogue	1	4	54	39354	13876	406	1	\N	dps	\N
2291	AB	AB6	Horde	Mage	1	1	26	9126	4680	372	1	\N	dps	\N
2292	AB	AB6	Alliance	Warrior	3	4	17	23676	9164	220	\N	1	dps	\N
2293	AB	AB6	Horde	Monk	0	2	46	2167	79019	570	1	\N	heal	\N
2294	AB	AB6	Horde	Paladin	6	3	54	72891	27209	558	1	\N	dps	\N
2295	AB	AB6	Alliance	Hunter	2	4	15	37034	6838	221	\N	1	dps	\N
2296	AB	AB6	Alliance	Warrior	4	7	18	37109	10036	233	\N	1	dps	\N
2297	AB	AB6	Alliance	Warrior	4	7	15	64983	17184	223	\N	1	dps	\N
2298	AB	AB6	Horde	Warrior	8	5	50	37346	5670	405	1	\N	dps	\N
2299	AB	AB6	Horde	Death Knight	12	2	51	56280	27106	558	1	\N	dps	\N
2300	AB	AB6	Alliance	Hunter	1	7	23	28445	7326	247	\N	1	dps	\N
2301	AB	AB6	Alliance	Paladin	0	4	8	18278	5566	177	\N	1	dps	\N
2302	AB	AB6	Horde	Rogue	5	1	52	45775	4281	570	1	\N	dps	\N
2303	AB	AB6	Horde	Demon Hunter	6	3	61	46891	4973	572	1	\N	dps	\N
2304	AB	AB6	Horde	Shaman	2	5	30	21703	8831	525	1	\N	dps	\N
2305	AB	AB6	Alliance	Rogue	1	2	19	15163	4293	229	\N	1	dps	\N
2306	BG	BG25	Alliance	Hunter	1	3	8	13629	7924	222	\N	1	dps	\N
2307	BG	BG25	Alliance	Warrior	2	5	5	24220	3029	212	\N	1	dps	\N
2308	BG	BG25	Alliance	Paladin	2	1	8	16994	4059	191	\N	1	dps	\N
2309	BG	BG25	Alliance	Rogue	1	0	8	12914	2010	191	\N	1	dps	\N
2310	BG	BG25	Horde	Paladin	3	0	20	39467	9094	510	1	\N	dps	\N
2311	BG	BG25	Alliance	Warlock	3	0	10	13059	6025	228	\N	1	dps	\N
2312	BG	BG25	Horde	Warlock	4	2	22	46862	16846	517	1	\N	dps	\N
2313	BG	BG25	Horde	Shaman	0	1	25	9653	55333	521	1	\N	heal	\N
2314	BG	BG25	Alliance	Druid	0	2	10	7933	2356	228	\N	1	dps	\N
2315	BG	BG25	Alliance	Paladin	0	4	8	2590	40707	224	\N	1	heal	\N
2316	BG	BG25	Horde	Death Knight	2	3	14	23149	13827	348	1	\N	dps	\N
2317	BG	BG25	Alliance	Paladin	0	1	7	12731	8011	204	\N	1	dps	\N
2318	BG	BG25	Horde	Hunter	3	2	21	18287	1046	515	1	\N	dps	\N
2319	BG	BG25	Horde	Hunter	1	1	16	12181	3478	351	1	\N	dps	\N
2320	BG	BG25	Horde	Priest	1	0	19	13207	1957	507	1	\N	dps	\N
2321	BG	BG25	Horde	Mage	4	1	22	37402	4834	514	1	\N	dps	\N
2322	BG	BG25	Horde	Shaman	4	0	18	28005	3528	355	1	\N	dps	\N
2323	BG	BG25	Horde	Hunter	4	1	16	16484	1920	353	1	\N	dps	\N
2324	ES	ES6	Alliance	Warrior	0	5	27	25807	37010	759	1	\N	dps	1
2325	ES	ES6	Alliance	Warlock	4	5	34	40809	20352	775	1	\N	dps	1
2326	ES	ES6	Alliance	Monk	3	4	28	27297	19112	1115	1	\N	dps	1
2327	ES	ES6	Alliance	Mage	0	7	28	47394	3058	1095	1	\N	dps	1
2328	ES	ES6	Horde	Rogue	4	1	41	41636	9457	362	\N	1	dps	1
2329	ES	ES6	Alliance	Druid	0	2	31	8216	118000	1119	1	\N	heal	1
2330	ES	ES6	Alliance	Paladin	0	4	20	3523	75378	743	1	\N	heal	1
2331	ES	ES6	Alliance	Demon Hunter	5	1	27	21088	3541	1076	1	\N	dps	1
2332	ES	ES6	Horde	Hunter	2	5	28	58681	5371	346	\N	1	dps	1
2333	ES	ES6	Horde	Warrior	5	4	40	66210	9229	364	\N	1	dps	1
2334	ES	ES6	Alliance	Monk	7	0	36	63343	28769	1134	1	\N	dps	1
2335	ES	ES6	Horde	Shaman	1	4	34	14851	92428	351	\N	1	heal	1
2336	ES	ES6	Horde	Hunter	3	1	43	40967	2553	358	\N	1	dps	1
2337	ES	ES6	Alliance	Demon Hunter	7	2	34	84422	4654	1130	1	\N	dps	1
2338	ES	ES6	Alliance	Warrior	1	3	27	44713	4474	1093	1	\N	dps	1
2339	ES	ES6	Alliance	Monk	3	5	24	49373	19907	764	1	\N	dps	1
2340	ES	ES6	Horde	Warrior	5	6	36	85790	7401	353	\N	1	dps	1
2341	ES	ES6	Horde	Druid	0	3	36	44771	33110	351	\N	1	dps	1
2342	ES	ES6	Horde	Shaman	2	4	35	32599	2259	361	\N	1	dps	1
2343	ES	ES6	Horde	Mage	3	3	48	102000	6822	368	\N	1	dps	1
2344	ES	ES6	Horde	Shaman	1	2	34	17472	69079	346	\N	1	heal	1
2345	ES	ES6	Alliance	Warlock	2	3	31	96175	22865	774	1	\N	dps	1
2346	ES	ES6	Horde	Mage	7	3	35	66191	9803	343	\N	1	dps	1
2347	ES	ES6	Horde	Priest	7	4	41	57182	16989	358	\N	1	dps	1
2348	ES	ES6	Horde	Rogue	3	3	28	46095	8750	320	\N	1	dps	1
2349	ES	ES6	Alliance	Priest	0	5	30	7184	74104	1115	1	\N	heal	1
2350	ES	ES6	Horde	Monk	0	1	46	3546	123000	370	\N	1	heal	1
2351	ES	ES6	Horde	Hunter	3	0	43	29871	3647	370	\N	1	dps	1
2352	ES	ES6	Alliance	Demon Hunter	5	4	31	75277	12759	1110	1	\N	dps	1
2353	ES	ES6	Alliance	Hunter	3	2	25	17413	7756	1103	1	\N	dps	1
2354	ES	ES7	Horde	Shaman	6	3	39	44741	11022	643	1	\N	dps	\N
2355	ES	ES7	Alliance	Demon Hunter	2	3	6	70538	7862	219	\N	1	dps	\N
2356	ES	ES7	Alliance	Hunter	1	5	11	33393	0	120	\N	1	dps	\N
2357	ES	ES7	Alliance	Warrior	1	3	15	19310	95	132	\N	1	dps	\N
2358	ES	ES7	Horde	Hunter	3	1	44	64569	5492	490	1	\N	dps	\N
2359	ES	ES7	Alliance	Druid	0	0	13	2208	46159	241	\N	1	heal	\N
2360	ES	ES7	Alliance	Demon Hunter	3	2	13	52064	6226	238	\N	1	dps	\N
2361	ES	ES7	Alliance	Druid	0	4	14	12	113000	243	\N	1	heal	\N
2362	ES	ES7	Alliance	Warlock	2	3	12	37793	9550	234	\N	1	dps	\N
2363	ES	ES7	Alliance	Monk	0	9	5	22768	11613	214	\N	1	dps	\N
2364	ES	ES7	Horde	Rogue	3	1	41	37217	3241	635	1	\N	dps	\N
2365	ES	ES7	Alliance	Mage	1	5	13	47364	15434	239	\N	1	dps	\N
2366	ES	ES7	Alliance	Paladin	2	1	15	8218	79470	245	\N	1	heal	\N
2367	ES	ES7	Alliance	Druid	0	2	8	3344	0	220	\N	1	dps	\N
2368	ES	ES7	Horde	Shaman	2	0	37	9153	110000	635	1	\N	heal	\N
2369	ES	ES7	Alliance	Shaman	1	3	12	23407	12949	236	\N	1	dps	\N
2370	ES	ES7	Horde	Hunter	5	1	42	63971	3709	628	1	\N	dps	\N
2371	ES	ES7	Horde	Priest	4	0	44	25209	88107	637	1	\N	heal	\N
2372	ES	ES7	Horde	Druid	5	0	38	22077	4263	627	1	\N	dps	\N
2373	ES	ES7	Horde	Hunter	1	0	26	18026	1367	468	1	\N	dps	\N
2374	ES	ES7	Horde	Druid	1	1	36	46841	116435	627	1	\N	heal	\N
2375	ES	ES7	Horde	Shaman	4	2	43	62416	22082	483	1	\N	dps	\N
2376	ES	ES7	Horde	Paladin	0	0	32	11490	9930	623	1	\N	dps	\N
2377	ES	ES7	Alliance	Demon Hunter	0	5	4	21706	5333	210	\N	1	dps	\N
2378	ES	ES7	Horde	Hunter	2	4	37	29457	8113	627	1	\N	dps	\N
2379	ES	ES7	Horde	Priest	0	0	45	14165	72554	486	1	\N	heal	\N
2380	ES	ES7	Horde	Druid	4	1	33	42632	4421	623	1	\N	dps	\N
2381	ES	ES7	Alliance	Hunter	0	5	16	65765	10051	247	\N	1	dps	\N
2382	ES	ES7	Horde	Demon Hunter	4	2	35	56060	5464	623	1	\N	dps	\N
2383	SM	SM16	Alliance	Druid	10	0	52	72753	5814	715	1	\N	dps	1
2384	SM	SM16	Alliance	Mage	6	0	51	61626	12396	1048	1	\N	dps	1
2385	SM	SM16	Horde	Druid	0	8	7	50999	2698	234	\N	1	dps	1
2386	SM	SM16	Horde	Shaman	0	9	9	39803	16304	238	\N	1	dps	1
2387	SM	SM16	Horde	Mage	3	3	11	52722	8052	243	\N	1	dps	1
2388	SM	SM16	Horde	Hunter	3	2	13	36993	6917	247	\N	1	dps	1
2389	SM	SM16	Alliance	Priest	5	1	47	65778	10115	701	1	\N	dps	1
2390	SM	SM16	Alliance	Death Knight	3	3	47	68523	33472	1042	1	\N	dps	1
2391	SM	SM16	Horde	Druid	0	6	11	1210	44721	243	\N	1	heal	1
2392	SM	SM16	Horde	Shaman	0	8	7	5301	53696	234	\N	1	heal	1
2393	SM	SM16	Horde	Druid	0	4	11	48433	30973	242	\N	1	dps	1
2394	SM	SM16	Alliance	Druid	0	2	48	0	83586	1044	1	\N	heal	1
2395	SM	SM16	Horde	Demon Hunter	3	8	6	51910	8760	232	\N	1	dps	1
2396	SM	SM16	Horde	Paladin	2	4	13	34762	12522	247	\N	1	dps	1
2397	SM	SM16	Horde	Mage	2	4	11	52423	14434	243	\N	1	dps	1
2398	SM	SM16	Alliance	Paladin	7	0	48	34276	17892	1041	1	\N	dps	1
2399	SM	SM16	Alliance	Monk	0	2	49	48	112000	1044	1	\N	heal	1
2400	SM	SM16	Alliance	Warrior	11	1	50	53914	8231	1047	1	\N	dps	1
2401	SM	SM16	Alliance	Monk	7	1	46	68868	18695	699	1	\N	dps	1
2402	SM	SM16	Alliance	Rogue	6	3	40	42804	1786	1036	1	\N	dps	1
2403	SM	SM17	Horde	Mage	0	5	20	48901	14294	431	1	\N	dps	1
2404	SM	SM17	Horde	Warrior	6	6	20	67083	9600	431	1	\N	dps	1
2405	SM	SM17	Alliance	Shaman	1	2	35	53763	6517	484	\N	1	dps	1
2406	SM	SM17	Horde	Demon Hunter	3	2	20	55679	8492	656	1	\N	dps	1
2407	SM	SM17	Horde	Monk	0	1	22	1482	173000	660	1	\N	heal	1
2408	SM	SM17	Horde	Shaman	0	1	13	5195	30084	358	1	\N	heal	1
2409	SM	SM17	Alliance	Hunter	1	5	28	86633	13275	471	\N	1	dps	1
2410	SM	SM17	Alliance	Druid	7	3	34	87272	21339	484	\N	1	dps	1
2411	SM	SM17	Alliance	Priest	0	3	34	24738	114000	481	\N	1	heal	1
2412	SM	SM17	Alliance	Death Knight	4	1	32	67247	22326	475	\N	1	dps	1
2413	SM	SM17	Horde	Monk	0	5	18	521	138000	427	1	\N	heal	1
2414	SM	SM17	Horde	Paladin	4	7	20	61329	24079	657	1	\N	dps	1
2415	SM	SM17	Alliance	Paladin	1	4	24	28042	6379	457	\N	1	dps	1
2416	SM	SM17	Alliance	Priest	7	2	34	114000	24281	487	\N	1	dps	1
2417	SM	SM17	Alliance	Paladin	4	0	36	60125	25773	319	\N	1	dps	1
2418	SM	SM17	Horde	Rogue	1	1	11	53876	6076	353	1	\N	dps	1
2419	SM	SM17	Horde	Mage	1	4	19	41051	9449	430	1	\N	dps	1
2420	SM	SM17	Alliance	Priest	12	0	33	74962	9774	263	\N	1	dps	1
2421	SM	SM17	Horde	Shaman	8	4	21	92886	18424	434	1	\N	dps	1
2422	SM	SM17	Alliance	Shaman	0	3	26	16344	109000	459	\N	1	heal	1
2423	TP	TP13	Alliance	Rogue	1	1	15	33634	9288	288	\N	1	dps	1
2424	TP	TP13	Horde	Shaman	3	3	14	29939	2179	664	1	\N	dps	1
2425	TP	TP13	Alliance	Warlock	1	6	11	12371	12542	271	\N	1	dps	1
2426	TP	TP13	Horde	Demon Hunter	5	0	18	27661	3076	672	1	\N	dps	1
2427	TP	TP13	Alliance	Mage	2	2	9	29025	12012	262	\N	1	dps	1
2428	TP	TP13	Horde	Shaman	0	1	19	6196	57026	672	1	\N	heal	1
2429	TP	TP13	Alliance	Paladin	1	3	8	42415	10479	248	\N	1	dps	1
2430	TP	TP13	Alliance	Priest	0	0	8	4669	53687	256	\N	1	heal	1
2431	TP	TP13	Alliance	Warlock	1	2	9	17772	11938	262	\N	1	dps	1
2432	TP	TP13	Alliance	Warlock	1	3	7	10812	2314	253	\N	1	dps	1
2433	TP	TP13	Alliance	Priest	3	4	10	47901	15828	267	\N	1	dps	1
2434	TP	TP13	Horde	Warrior	2	1	17	32212	4479	448	1	\N	dps	1
2435	TP	TP13	Horde	Warrior	4	2	21	29256	5581	452	1	\N	dps	1
2436	TP	TP13	Alliance	Paladin	3	4	12	53936	12962	275	\N	1	dps	1
2437	TP	TP13	Horde	Warlock	2	3	11	19341	8754	428	1	\N	dps	1
2438	TP	TP13	Horde	Priest	4	2	16	33770	6774	665	1	\N	dps	1
2439	TP	TP13	Horde	Warlock	4	1	16	70233	18003	671	1	\N	dps	1
2440	TP	TP13	Horde	Demon Hunter	4	1	19	38725	6528	671	1	\N	dps	1
2441	TP	TP13	Alliance	Rogue	2	3	7	24556	6083	256	\N	1	dps	1
2442	TP	TP13	Horde	Priest	0	1	22	12802	93809	456	1	\N	heal	1
2443	TP	TP14	Alliance	Warlock	2	3	21	78637	31652	951	1	\N	dps	1
2444	TP	TP14	Horde	Shaman	6	3	42	58509	5291	271	\N	1	dps	1
2445	TP	TP14	Horde	Warlock	7	1	33	98288	46397	256	\N	1	dps	1
2446	TP	TP14	Horde	Druid	0	1	43	7963	118000	274	\N	1	heal	1
2447	TP	TP14	Alliance	Death Knight	1	4	13	16768	12760	915	1	\N	dps	1
2448	TP	TP14	Horde	Warrior	18	0	45	86247	11423	279	\N	1	dps	1
2449	TP	TP14	Horde	Shaman	1	5	33	11183	78644	259	\N	1	heal	1
2450	TP	TP14	Horde	Warrior	3	4	22	19488	13092	236	\N	1	dps	1
2451	TP	TP14	Horde	Mage	1	7	18	41750	3122	223	\N	1	dps	1
2452	TP	TP14	Alliance	Rogue	6	5	20	83211	14177	618	1	\N	dps	1
2453	TP	TP14	Alliance	Warrior	4	10	13	61479	12604	584	1	\N	dps	1
2454	TP	TP14	Horde	Demon Hunter	7	2	28	60787	15181	248	\N	1	dps	1
2455	TP	TP14	Alliance	Hunter	3	6	16	61370	7121	592	1	\N	dps	1
2456	TP	TP14	Alliance	Druid	0	6	12	1663	58244	910	1	\N	heal	1
2457	TP	TP14	Alliance	Priest	0	6	11	2747	57114	905	1	\N	heal	1
2458	TP	TP14	Alliance	Rogue	3	6	14	46336	8072	909	1	\N	dps	1
2459	TP	TP14	Horde	Hunter	7	1	39	88351	5483	269	\N	1	dps	1
2460	TP	TP14	Horde	Druid	5	3	44	59769	6680	279	\N	1	dps	1
2461	TP	TP14	Alliance	Priest	2	5	10	37080	25063	569	1	\N	dps	1
2462	TP	TP14	Alliance	Warrior	6	4	17	56107	17794	934	1	\N	dps	1
2463	TP	TP15	Horde	Mage	3	3	21	24718	6235	527	1	\N	dps	\N
2464	TP	TP15	Horde	Demon Hunter	2	1	30	17887	2230	545	1	\N	dps	\N
2465	TP	TP15	Alliance	Monk	1	3	16	18622	16727	238	\N	1	dps	\N
2466	TP	TP15	Horde	Monk	0	2	21	684	28731	537	1	\N	heal	\N
2467	TP	TP15	Horde	Shaman	0	1	31	9907	59273	395	1	\N	heal	\N
2468	TP	TP15	Horde	Death Knight	6	2	28	31709	7046	393	1	\N	dps	\N
2469	TP	TP15	Alliance	Demon Hunter	0	4	9	12380	873	214	\N	1	dps	\N
2470	TP	TP15	Horde	Warrior	6	4	26	54586	5175	538	1	\N	dps	\N
2471	TP	TP15	Alliance	Rogue	0	3	14	23328	4003	231	\N	1	dps	\N
2472	TP	TP15	Alliance	Shaman	0	4	3	13488	0	182	\N	1	dps	\N
2473	TP	TP15	Horde	Demon Hunter	3	2	29	22816	2599	394	1	\N	dps	\N
2474	TP	TP15	Alliance	Hunter	1	6	15	40247	10384	235	\N	1	dps	\N
2475	TP	TP15	Alliance	Death Knight	5	2	10	33673	20513	202	\N	1	dps	\N
2476	TP	TP15	Alliance	Druid	0	3	8	51	36780	210	\N	1	heal	\N
2477	TP	TP15	Alliance	Monk	3	4	13	27462	16376	226	\N	1	dps	\N
2478	TP	TP15	Horde	Monk	3	0	28	42135	12250	392	1	\N	dps	\N
2479	TP	TP15	Alliance	Mage	3	4	12	25968	3387	225	\N	1	dps	\N
2480	TP	TP15	Horde	Rogue	7	0	32	52279	4973	400	1	\N	dps	\N
2481	TP	TP15	Horde	Priest	7	1	34	58823	16290	402	1	\N	dps	\N
2482	TP	TP15	Alliance	Warlock	3	3	14	35167	23944	232	\N	1	dps	\N
2483	TP	TP16	Horde	Mage	4	4	22	30152	3995	193	\N	1	dps	\N
2484	TP	TP16	Alliance	Paladin	7	4	23	45938	16191	487	1	\N	dps	\N
2485	TP	TP16	Alliance	Warrior	1	6	16	28822	0	689	1	\N	dps	\N
2486	TP	TP16	Horde	Rogue	1	3	14	31461	3521	151	\N	1	dps	\N
2487	TP	TP16	Horde	Demon Hunter	7	3	25	50706	5248	198	\N	1	dps	\N
2488	TP	TP16	Alliance	Hunter	1	4	22	16091	6837	706	1	\N	dps	\N
2489	TP	TP16	Horde	Druid	2	3	20	36005	4733	186	\N	1	dps	\N
2490	TP	TP16	Alliance	Mage	3	4	30	32858	2058	506	1	\N	dps	\N
2491	TP	TP16	Alliance	Paladin	8	5	27	45134	12837	724	1	\N	dps	\N
2492	TP	TP16	Alliance	Death Knight	1	0	26	55732	23685	502	1	\N	dps	\N
2493	TP	TP16	Horde	Shaman	1	2	30	15862	89427	210	\N	1	heal	\N
2494	TP	TP16	Alliance	Priest	2	1	27	6066	55883	500	1	\N	heal	\N
2495	TP	TP16	Alliance	Warrior	5	3	30	75330	26564	509	1	\N	dps	\N
2496	TP	TP16	Horde	Shaman	1	7	12	23979	18794	170	\N	1	dps	\N
2497	TP	TP16	Alliance	Druid	2	0	31	14984	73617	737	1	\N	heal	\N
2498	TP	TP16	Horde	Warlock	5	3	25	34091	17165	197	\N	1	dps	\N
2499	TP	TP16	Alliance	Mage	5	6	19	60797	9788	478	1	\N	dps	\N
2500	TP	TP16	Horde	Hunter	4	4	24	32469	9885	194	\N	1	dps	\N
2501	TP	TP16	Horde	Paladin	0	5	15	95502	31823	174	\N	1	dps	\N
2502	TP	TP16	Horde	Paladin	8	1	28	46024	22912	203	\N	1	dps	\N
2503	BG	BG26	Horde	Shaman	0	2	14	2579	65173	352	1	\N	heal	\N
2504	BG	BG26	Horde	Monk	1	2	17	154	59401	355	1	\N	heal	\N
2505	BG	BG26	Alliance	Warlock	3	6	8	103000	35838	217	\N	1	dps	\N
2506	BG	BG26	Horde	Paladin	4	1	15	40138	23324	354	1	\N	dps	\N
2507	BG	BG26	Alliance	Death Knight	1	3	7	17164	18384	219	\N	1	dps	\N
2508	BG	BG26	Alliance	Druid	4	3	12	49808	14442	240	\N	1	dps	\N
2509	BG	BG26	Alliance	Rogue	1	3	10	20978	12313	228	\N	1	dps	\N
2510	BG	BG26	Horde	Paladin	5	0	19	34984	12045	517	1	\N	dps	\N
2511	BG	BG26	Horde	Demon Hunter	5	4	11	55357	10174	501	1	\N	dps	\N
2512	BG	BG26	Horde	Paladin	2	1	19	33555	9599	362	1	\N	dps	\N
2513	BG	BG26	Horde	Warlock	1	4	4	18378	8630	335	1	\N	dps	\N
2514	BG	BG26	Horde	Hunter	2	0	19	34062	6011	372	1	\N	dps	\N
2515	BG	BG26	Horde	Mage	5	1	19	44478	5243	517	1	\N	dps	\N
2516	BG	BG26	Horde	Druid	1	0	2	9029	4538	327	1	\N	dps	\N
2517	BG	BG26	Alliance	Rogue	1	3	7	13179	1791	216	\N	1	dps	\N
2518	BG	BG27	Alliance	Hunter	0	1	7	21442	5437	222	\N	1	dps	\N
2519	BG	BG27	Alliance	Demon Hunter	0	3	6	30948	13084	219	\N	1	dps	\N
2520	BG	BG27	Horde	Warrior	0	0	18	8708	700	208	1	\N	dps	\N
2521	BG	BG27	Horde	Rogue	1	0	1	1543	0	473	1	\N	dps	\N
2522	BG	BG27	Horde	Mage	3	0	24	38154	4869	520	1	\N	dps	\N
2523	BG	BG27	Horde	Death Knight	3	2	25	60011	13190	521	1	\N	dps	\N
2524	BG	BG27	Alliance	Death Knight	2	1	7	22166	7439	222	\N	1	dps	\N
2525	BG	BG27	Horde	Shaman	0	2	25	6454	90365	371	1	\N	heal	\N
2526	BG	BG27	Horde	Mage	1	0	25	26806	7064	521	1	\N	dps	\N
2527	BG	BG27	Horde	Warrior	7	0	24	28819	7315	370	1	\N	dps	\N
2528	BG	BG27	Horde	Warlock	6	2	24	68216	17551	518	1	\N	dps	\N
2529	BG	BG27	Alliance	Death Knight	0	2	5	12355	7194	215	\N	1	dps	\N
2530	BG	BG27	Alliance	Warrior	2	6	5	21008	11092	215	\N	1	dps	\N
2531	BG	BG27	Alliance	Warlock	1	4	7	45357	17826	222	\N	1	dps	\N
2532	BG	BG27	Alliance	Paladin	2	5	7	19143	13923	222	\N	1	dps	\N
2533	BG	BG27	Horde	Hunter	3	1	24	33482	6872	368	1	\N	dps	\N
2534	BG	BG27	Alliance	Druid	0	3	7	0	69910	222	\N	1	heal	\N
2535	BG	BG27	Horde	Warlock	2	0	25	16745	4845	521	1	\N	dps	\N
2536	AB	AB7	Horde	Rogue	4	0	46	27239	1068	425	1	\N	dps	\N
2537	AB	AB7	Horde	Rogue	5	3	40	34711	4836	546	1	\N	dps	\N
2538	AB	AB7	Alliance	Druid	1	7	8	37473	6613	242	\N	1	dps	\N
2539	AB	AB7	Horde	Demon Hunter	1	0	40	53847	54700	391	1	\N	dps	\N
2540	AB	AB7	Alliance	Priest	6	3	8	50761	17496	245	\N	1	dps	\N
2541	AB	AB7	Horde	Hunter	5	2	41	47827	8017	392	1	\N	dps	\N
2542	AB	AB7	Alliance	Mage	1	2	5	20397	4430	235	\N	1	dps	\N
2543	AB	AB7	Horde	Shaman	8	2	30	52754	2307	384	1	\N	dps	\N
2544	AB	AB7	Horde	Mage	6	0	37	55584	16291	538	1	\N	dps	\N
2545	AB	AB7	Horde	Shaman	0	0	39	9206	56252	537	1	\N	heal	\N
2546	AB	AB7	Alliance	Priest	2	4	14	55752	16909	266	\N	1	dps	\N
2547	AB	AB7	Horde	Rogue	1	0	3	6170	117	361	1	\N	dps	\N
2548	AB	AB7	Alliance	Death Knight	1	5	12	89171	30882	262	\N	1	dps	\N
2549	AB	AB7	Alliance	Warlock	0	7	9	14427	1634	248	\N	1	dps	\N
2550	AB	AB7	Alliance	Warrior	0	2	5	11550	628	232	\N	1	dps	\N
2551	AB	AB7	Alliance	Monk	2	3	5	38716	30693	243	\N	1	dps	\N
2552	AB	AB7	Horde	Mage	2	1	39	27270	6213	559	1	\N	dps	\N
2553	AB	AB7	Horde	Monk	4	0	19	20296	11997	354	1	\N	dps	\N
2554	AB	AB7	Alliance	Mage	0	6	1	10966	1657	221	\N	1	dps	\N
2555	AB	AB7	Alliance	Shaman	0	5	12	1939	50584	256	\N	1	heal	\N
2556	AB	AB7	Horde	Priest	1	0	36	21095	66086	390	1	\N	heal	\N
2557	AB	AB7	Horde	Rogue	4	3	26	17277	2267	375	1	\N	dps	\N
2558	AB	AB7	Horde	Priest	0	1	40	20178	96774	242	1	\N	heal	\N
2559	AB	AB7	Horde	Warrior	8	3	33	71695	18676	535	1	\N	dps	\N
2560	AB	AB7	Horde	Druid	3	3	30	56667	15319	370	1	\N	dps	\N
2561	AB	AB7	Alliance	Hunter	0	0	6	13533	662	189	\N	1	dps	\N
2562	AB	AB7	Alliance	Warlock	2	6	12	37450	21536	259	\N	1	dps	\N
2563	AB	AB8	Alliance	Hunter	0	2	18	30433	2460	160	\N	1	dps	\N
2564	AB	AB8	Horde	Hunter	1	0	43	13557	83	432	1	\N	dps	\N
2565	AB	AB8	Alliance	Mage	5	5	33	61371	7492	330	\N	1	dps	\N
2566	AB	AB8	Horde	Death Knight	6	7	25	86053	32145	361	1	\N	dps	\N
2567	AB	AB8	Alliance	Death Knight	6	3	33	81815	18054	337	\N	1	dps	\N
2568	AB	AB8	Alliance	Shaman	0	2	31	376	74502	320	\N	1	heal	\N
2569	AB	AB8	Horde	Rogue	1	4	5	32901	4406	483	1	\N	dps	\N
2570	AB	AB8	Alliance	Mage	5	5	32	59780	6497	337	\N	1	dps	\N
2571	AB	AB8	Horde	Shaman	5	4	39	81232	4843	550	1	\N	dps	\N
2572	AB	AB8	Horde	Mage	6	4	40	85971	7710	550	1	\N	dps	\N
2573	AB	AB8	Alliance	Warrior	0	1	22	14625	151	179	\N	1	dps	\N
2574	AB	AB8	Horde	Shaman	13	1	46	152000	19158	418	1	\N	dps	\N
2575	AB	AB8	Horde	Shaman	0	5	21	18065	134000	357	1	\N	heal	\N
2576	AB	AB8	Alliance	Druid	5	8	31	72873	26771	317	\N	1	dps	\N
2577	AB	AB8	Alliance	Druid	0	2	30	14989	142000	341	\N	1	heal	\N
2578	AB	AB8	Horde	Paladin	2	4	44	47024	34247	567	1	\N	dps	\N
2579	AB	AB8	Horde	Shaman	0	1	42	5041	39188	581	1	\N	heal	\N
2580	AB	AB8	Alliance	Paladin	4	8	25	63910	12486	308	\N	1	dps	\N
2581	AB	AB8	Horde	Priest	0	3	37	11169	121000	549	1	\N	heal	\N
2582	AB	AB8	Alliance	Druid	2	3	21	18304	94497	315	\N	1	heal	\N
2583	AB	AB8	Alliance	Mage	2	3	23	68646	955	320	\N	1	dps	\N
2584	AB	AB8	Horde	Rogue	4	3	29	57225	2269	547	1	\N	dps	\N
2585	AB	AB8	Alliance	Rogue	5	5	24	68246	5319	340	\N	1	dps	\N
2586	AB	AB8	Alliance	Mage	12	0	28	72836	4575	209	\N	1	dps	\N
2587	AB	AB8	Alliance	Priest	0	2	39	21738	178000	360	\N	1	heal	\N
2588	AB	AB8	Horde	Warlock	4	3	28	65517	34603	527	1	\N	dps	\N
2589	AB	AB8	Horde	Druid	11	5	42	105000	7269	553	1	\N	dps	\N
2590	AB	AB8	Horde	Warlock	1	5	25	44523	33383	513	1	\N	dps	\N
2591	AB	AB8	Alliance	Paladin	2	3	33	62039	26159	342	\N	1	dps	\N
2592	AB	AB8	Horde	Rogue	1	1	13	45638	8349	493	1	\N	dps	\N
2593	SM	SM18	Alliance	Warrior	9	6	31	64878	12036	571	1	\N	dps	\N
2594	SM	SM18	Alliance	Hunter	1	5	31	46972	10527	563	1	\N	dps	\N
2595	SM	SM18	Alliance	Druid	4	7	28	45914	15405	776	1	\N	dps	\N
2596	SM	SM18	Alliance	Paladin	2	6	29	44282	21805	560	1	\N	dps	\N
2597	SM	SM18	Horde	Priest	12	2	55	69768	25886	280	\N	1	dps	\N
2598	SM	SM18	Alliance	Druid	0	5	30	40268	12248	560	1	\N	dps	\N
2599	SM	SM18	Horde	Shaman	1	4	51	11681	120000	272	\N	1	heal	\N
2600	SM	SM18	Alliance	Warrior	11	7	31	47626	10030	788	1	\N	dps	\N
2601	SM	SM18	Horde	Warlock	11	3	54	54964	21751	278	\N	1	dps	\N
2602	SM	SM18	Horde	Warlock	14	2	56	54636	21335	282	\N	1	dps	\N
2603	SM	SM18	Alliance	Rogue	4	6	35	59232	16592	804	1	\N	dps	\N
2604	SM	SM18	Alliance	Paladin	1	9	35	75366	18664	803	1	\N	dps	\N
2605	SM	SM18	Horde	Mage	2	5	45	41122	15635	184	\N	1	dps	\N
2606	SM	SM18	Alliance	Hunter	3	1	35	33914	8201	804	1	\N	dps	\N
2607	SM	SM18	Horde	Mage	1	5	47	23531	21890	266	\N	1	dps	\N
2608	SM	SM18	Horde	Warlock	2	3	33	12778	13420	228	\N	1	dps	\N
2609	SM	SM18	Horde	Paladin	6	2	49	76370	24944	269	\N	1	dps	\N
2610	SM	SM18	Horde	Mage	2	7	45	46803	8438	262	\N	1	dps	\N
2611	SM	SM18	Alliance	Warrior	4	6	34	28233	4772	802	1	\N	dps	\N
2612	SM	SM18	Horde	Shaman	6	5	52	55160	11923	276	\N	1	dps	\N
2613	SM	SM19	Alliance	Priest	0	2	19	5360	57970	241	\N	1	heal	\N
2614	SM	SM19	Alliance	Shaman	4	2	19	67727	5071	238	\N	1	dps	\N
2615	SM	SM19	Horde	Mage	3	4	13	56374	11710	489	1	\N	dps	\N
2616	SM	SM19	Alliance	Mage	1	2	20	40253	13253	240	\N	1	dps	\N
2617	SM	SM19	Horde	Demon Hunter	0	2	18	41053	29495	499	1	\N	dps	\N
2618	SM	SM19	Alliance	Warlock	6	1	19	76237	34200	239	\N	1	dps	\N
2619	SM	SM19	Horde	Warrior	3	2	17	26617	680	347	1	\N	dps	\N
2620	SM	SM19	Alliance	Warrior	1	1	4	11503	1244	133	\N	1	dps	\N
2621	SM	SM19	Horde	Warlock	1	1	19	40794	9038	351	1	\N	dps	\N
2622	SM	SM19	Horde	Priest	0	1	17	7283	98892	497	1	\N	heal	\N
2623	SM	SM19	Horde	Warrior	3	3	12	22933	2225	337	1	\N	dps	\N
2624	SM	SM19	Alliance	Priest	5	2	21	80772	11237	245	\N	1	dps	\N
2625	SM	SM19	Horde	Shaman	0	2	18	10988	84513	499	1	\N	heal	\N
2626	SM	SM19	Alliance	Priest	0	2	20	27621	123000	242	\N	1	heal	\N
2627	SM	SM19	Alliance	Paladin	0	2	21	14924	9505	244	\N	1	dps	\N
2628	SM	SM19	Alliance	Warlock	0	1	3	2819	1668	133	\N	1	dps	\N
2629	SM	SM19	Alliance	Druid	4	4	18	43314	14914	235	\N	1	dps	\N
2630	SM	SM19	Horde	Warrior	5	0	19	26456	7376	351	1	\N	dps	\N
2631	SM	SM19	Horde	Hunter	1	2	19	83616	2853	351	1	\N	dps	\N
2632	SM	SM19	Horde	Warrior	3	5	14	40242	6804	341	1	\N	dps	\N
2633	TK	TK18	Horde	Druid	2	8	18	46023	3560	236	\N	1	dps	\N
2634	TK	TK18	Alliance	Mage	3	2	54	50139	11408	717	1	\N	dps	\N
2635	TK	TK18	Alliance	Monk	2	4	49	19682	63722	1038	1	\N	heal	\N
2636	TK	TK18	Horde	Shaman	0	5	18	8155	52525	237	\N	1	heal	\N
2637	TK	TK18	Alliance	Mage	5	5	44	31525	14035	688	1	\N	dps	\N
2638	TK	TK18	Alliance	Hunter	8	1	60	74246	5960	1069	1	\N	dps	\N
2639	TK	TK18	Horde	Mage	1	5	17	64917	19161	235	\N	1	dps	\N
2640	TK	TK18	Alliance	Rogue	2	5	48	30889	7506	695	1	\N	dps	\N
2641	TK	TK18	Alliance	Paladin	3	1	59	67543	20838	1068	1	\N	dps	\N
2642	TK	TK18	Horde	Death Knight	2	6	18	79421	31183	237	\N	1	dps	\N
2643	TK	TK18	Horde	Warrior	6	7	13	45738	3546	227	\N	1	dps	\N
2644	TK	TK18	Alliance	Priest	0	1	59	1637	195000	731	1	\N	heal	\N
2645	TK	TK18	Horde	Demon Hunter	0	7	14	44186	10973	229	\N	1	dps	\N
2646	TK	TK18	Horde	Mage	3	7	16	66487	9653	231	\N	1	dps	\N
2647	TK	TK18	Alliance	Hunter	4	2	56	62711	3727	1060	1	\N	dps	\N
2648	TK	TK18	Alliance	Warlock	9	0	62	87828	27487	1074	1	\N	dps	\N
2649	TK	TK18	Alliance	Paladin	23	0	62	101000	12423	737	1	\N	dps	\N
2650	TK	TK18	Horde	Death Knight	2	4	17	25122	18828	235	\N	1	dps	\N
2651	TK	TK18	Horde	Paladin	0	6	17	10283	46352	235	\N	1	heal	\N
2652	TK	TK18	Horde	Warlock	5	7	15	48000	16814	231	\N	1	dps	\N
2653	TK	TK19	Alliance	Demon Hunter	3	2	47	70484	10373	580	1	\N	dps	\N
2654	TK	TK19	Horde	Rogue	7	4	26	47437	4763	201	\N	1	dps	\N
2655	TK	TK19	Alliance	Druid	0	2	47	1014	128000	814	1	\N	heal	\N
2656	TK	TK19	Alliance	Paladin	5	4	42	46811	24846	794	1	\N	dps	\N
2657	TK	TK19	Alliance	Warlock	9	3	49	81937	15134	818	1	\N	dps	\N
2658	TK	TK19	Alliance	Mage	4	4	45	33576	12256	805	1	\N	dps	\N
2659	TK	TK19	Alliance	Mage	6	2	49	76287	6535	592	1	\N	dps	\N
2660	TK	TK19	Horde	Mage	3	4	24	71009	3386	197	\N	1	dps	\N
2661	TK	TK19	Horde	Shaman	1	5	22	11168	86090	193	\N	1	heal	\N
2662	TK	TK19	Alliance	Demon Hunter	4	3	43	26973	22460	801	1	\N	dps	\N
2663	TK	TK19	Horde	Warrior	6	6	25	52467	9495	199	\N	1	dps	\N
2664	TK	TK19	Horde	Demon Hunter	1	5	19	29207	2983	187	\N	1	dps	\N
2665	TK	TK19	Horde	Priest	0	5	23	10330	114000	195	\N	1	heal	\N
2666	TK	TK19	Alliance	Demon Hunter	10	1	50	87765	10367	815	1	\N	dps	\N
2667	TK	TK19	Horde	Demon Hunter	5	4	24	66599	8955	197	\N	1	dps	\N
2668	TK	TK19	Horde	Shaman	3	8	21	66610	6397	191	\N	1	dps	\N
2669	TK	TK19	Alliance	Demon Hunter	4	3	51	67876	8942	825	1	\N	dps	\N
2670	TK	TK19	Alliance	Shaman	0	5	46	4639	60170	585	1	\N	heal	\N
2671	TK	TK19	Horde	Mage	3	6	22	58897	15577	193	\N	1	dps	\N
2672	TK	TK19	Horde	Warlock	0	2	29	40329	13640	207	\N	1	dps	\N
2673	TK	TK20	Horde	Warlock	2	3	22	43576	20655	185	\N	1	dps	\N
2674	TK	TK20	Alliance	Warrior	3	6	36	19329	3194	561	1	\N	dps	\N
2675	TK	TK20	Alliance	Paladin	0	5	39	3606	98315	794	1	\N	heal	\N
2676	TK	TK20	Alliance	Warlock	7	4	40	85586	26630	800	1	\N	dps	\N
2677	TK	TK20	Horde	Rogue	5	7	37	75885	14350	250	\N	1	dps	\N
2678	TK	TK20	Horde	Death Knight	4	5	38	95537	86042	252	\N	1	dps	\N
2679	TK	TK20	Horde	Mage	8	3	38	77197	7210	252	\N	1	dps	\N
2680	TK	TK20	Horde	Warrior	3	5	38	52652	4186	252	\N	1	dps	\N
2681	TK	TK20	Horde	Shaman	0	8	36	7988	93927	248	\N	1	heal	\N
2682	TK	TK20	Alliance	Paladin	10	3	36	96948	20301	771	1	\N	dps	\N
2683	TK	TK20	Horde	Hunter	3	3	36	40886	11454	248	\N	1	dps	\N
2684	TK	TK20	Alliance	Paladin	15	1	42	120000	36123	579	1	\N	dps	\N
2685	TK	TK20	Alliance	Monk	0	5	41	35008	61404	801	1	\N	heal	\N
2686	TK	TK20	Alliance	Rogue	3	4	41	58569	12269	800	1	\N	dps	\N
2687	TK	TK20	Horde	Monk	0	3	41	6639	116000	258	\N	1	heal	\N
2688	TK	TK20	Alliance	Druid	2	5	41	52062	17833	577	1	\N	dps	\N
2689	TK	TK20	Horde	Hunter	8	3	31	47878	10866	213	\N	1	dps	\N
2690	TK	TK20	Alliance	Paladin	2	5	37	92763	23055	792	1	\N	dps	\N
2691	TK	TK20	Horde	Mage	5	3	39	86515	16473	254	\N	1	dps	\N
2692	TK	TK20	Alliance	Hunter	2	6	41	57969	9310	577	1	\N	dps	\N
2693	TK	TK21	Alliance	Monk	4	4	22	39702	19076	246	\N	1	dps	\N
2694	TK	TK21	Horde	Rogue	6	2	38	42395	4749	529	1	\N	dps	\N
2695	TK	TK21	Horde	Warlock	1	5	34	19102	12146	521	1	\N	dps	\N
2696	TK	TK21	Alliance	Paladin	3	5	23	52063	14630	252	\N	1	dps	\N
2697	TK	TK21	Horde	Death Knight	4	4	33	39191	18903	370	1	\N	dps	\N
2698	TK	TK21	Horde	Shaman	0	4	41	10269	91595	536	1	\N	heal	\N
2699	TK	TK21	Horde	Death Knight	6	1	43	23545	14513	540	1	\N	dps	\N
2700	TK	TK21	Alliance	Warlock	5	4	22	56880	22975	246	\N	1	dps	\N
2701	TK	TK21	Horde	Warrior	7	3	38	38509	11850	379	1	\N	dps	\N
2702	TK	TK21	Alliance	Mage	2	6	22	31119	769	245	\N	1	dps	\N
2703	TK	TK21	Horde	Demon Hunter	1	4	29	11679	1956	362	1	\N	dps	\N
2704	TK	TK21	Alliance	Shaman	3	6	24	42817	6532	253	\N	1	dps	\N
2705	TK	TK21	Alliance	Druid	0	5	21	25909	4852	242	\N	1	dps	\N
2706	TK	TK21	Alliance	Warrior	5	4	24	42177	2774	251	\N	1	dps	\N
2707	TK	TK21	Horde	Druid	5	2	39	72724	11281	381	1	\N	dps	\N
2708	TK	TK21	Alliance	Paladin	0	3	27	3323	40559	264	\N	1	heal	\N
2709	TK	TK21	Horde	Mage	6	4	33	38118	7450	519	1	\N	dps	\N
2710	TK	TK21	Horde	Warlock	7	1	41	46512	8569	386	1	\N	dps	\N
2711	TK	TK21	Alliance	Warrior	7	3	20	29758	8636	231	\N	1	dps	\N
2712	TK	TK22	Horde	Mage	7	6	58	79443	40491	426	1	\N	dps	\N
2713	TK	TK22	Alliance	Mage	4	6	54	75880	11597	466	\N	1	dps	\N
2714	TK	TK22	Alliance	Druid	0	7	44	0	69591	434	\N	1	heal	\N
2715	TK	TK22	Horde	Rogue	5	5	53	34547	9505	416	1	\N	dps	\N
2716	TK	TK22	Alliance	Paladin	14	7	50	83601	20097	456	\N	1	dps	\N
2717	TK	TK22	Alliance	Shaman	5	8	47	73443	19092	445	\N	1	dps	\N
2718	TK	TK22	Alliance	Mage	4	5	46	52811	10124	444	\N	1	dps	\N
2719	TK	TK22	Horde	Shaman	0	7	56	10096	80033	422	1	\N	heal	\N
2720	TK	TK22	Alliance	Shaman	3	6	50	38351	13031	453	\N	1	dps	\N
2721	TK	TK22	Horde	Rogue	4	6	54	14411	21131	418	1	\N	dps	\N
2722	TK	TK22	Horde	Warlock	21	2	62	120000	98595	434	1	\N	dps	\N
2723	TK	TK22	Horde	Warrior	11	7	47	77200	34835	404	1	\N	dps	\N
2724	TK	TK22	Horde	Death Knight	4	6	51	50851	23812	562	1	\N	dps	\N
2725	TK	TK22	Horde	Warrior	5	7	44	26669	5950	548	1	\N	dps	\N
2726	TK	TK22	Horde	Rogue	4	7	52	53392	16724	414	1	\N	dps	\N
2727	TK	TK22	Alliance	Warlock	8	4	55	55898	17971	469	\N	1	dps	\N
2728	TK	TK22	Alliance	Death Knight	9	6	49	98756	22422	453	\N	1	dps	\N
2729	TK	TK22	Alliance	Hunter	5	6	49	73531	6765	453	\N	1	dps	\N
2730	TK	TK22	Alliance	Druid	7	7	50	91472	10294	453	\N	1	dps	\N
2731	TK	TK22	Horde	Hunter	1	8	49	36503	11728	558	1	\N	dps	\N
2732	WG	WG29	Alliance	Monk	0	2	46	273	212000	433	\N	1	heal	1
2733	WG	WG29	Alliance	Warrior	17	5	45	138000	22898	431	\N	1	dps	1
2734	WG	WG29	Horde	Rogue	5	3	28	69020	8216	638	1	\N	dps	1
2735	WG	WG29	Horde	Warrior	2	8	23	74238	7893	397	1	\N	dps	1
2736	WG	WG29	Horde	Shaman	1	2	24	4140	98154	617	1	\N	heal	1
2737	WG	WG29	Alliance	Mage	6	3	45	89714	18490	431	\N	1	dps	1
2738	WG	WG29	Horde	Shaman	0	5	18	14268	51726	591	1	\N	heal	1
2739	WG	WG29	Horde	Warlock	4	4	28	79250	29924	409	1	\N	dps	1
2740	WG	WG29	Alliance	Hunter	5	4	36	36925	9128	417	\N	1	dps	1
2741	WG	WG29	Horde	Warlock	8	5	25	130000	53295	629	1	\N	dps	1
2742	WG	WG29	Alliance	Hunter	3	4	43	73168	9238	427	\N	1	dps	1
2743	WG	WG29	Alliance	Demon Hunter	2	3	45	52055	1711	429	\N	1	dps	1
2744	WG	WG29	Horde	Demon Hunter	6	1	22	50977	8559	376	1	\N	dps	1
2745	WG	WG29	Horde	Demon Hunter	1	5	17	75031	8670	365	1	\N	dps	1
2746	WG	WG29	Horde	Paladin	1	2	25	4266	81144	621	1	\N	heal	1
2747	WG	WG29	Alliance	Warrior	4	4	44	60134	13751	433	\N	1	dps	1
2748	WG	WG29	Alliance	Rogue	8	3	48	80889	15353	442	\N	1	dps	1
2749	WG	WG30	Horde	Death Knight	1	0	14	37010	48361	511	1	\N	dps	\N
2750	WG	WG30	Alliance	Rogue	2	3	14	29309	15555	205	\N	1	dps	\N
2751	WG	WG30	Alliance	Mage	1	2	13	29756	10065	203	\N	1	dps	\N
2752	WG	WG30	Alliance	Monk	2	2	15	3208	30150	209	\N	1	heal	\N
2753	WG	WG30	Horde	Hunter	1	2	13	53986	6408	353	1	\N	dps	\N
2754	WG	WG30	Alliance	Death Knight	2	3	10	67980	10067	194	\N	1	dps	\N
2755	WG	WG30	Alliance	Hunter	2	1	14	47660	9277	205	\N	1	dps	\N
2756	WG	WG30	Horde	Hunter	1	1	10	10360	7126	289	1	\N	dps	\N
2757	WG	WG30	Horde	Druid	5	3	10	51450	7569	500	1	\N	dps	\N
2758	WG	WG30	Horde	Shaman	0	0	9	993	31357	428	1	\N	heal	\N
2759	WG	WG30	Horde	Mage	2	2	14	28486	7140	508	1	\N	dps	\N
2760	WG	WG30	Horde	Shaman	1	0	8	5893	25072	426	1	\N	heal	\N
2761	WG	WG30	Horde	Monk	0	1	11	2518	52178	502	1	\N	heal	\N
2762	WG	WG30	Alliance	Priest	0	1	15	26420	143000	209	\N	1	heal	\N
2763	WG	WG30	Horde	Hunter	3	1	14	64676	6194	357	1	\N	dps	\N
2764	WG	WG30	Alliance	Warrior	3	3	14	52009	10815	205	\N	1	dps	\N
2765	WG	WG30	Alliance	Death Knight	1	0	14	30151	16314	206	\N	1	dps	\N
2766	WG	WG30	Horde	Shaman	2	1	12	75036	4745	506	1	\N	dps	\N
2767	WG	WG30	Alliance	Priest	2	0	15	73141	18199	209	\N	1	dps	\N
2768	AB	AB9	Horde	Death Knight	3	4	56	68759	2003	440	1	\N	dps	\N
2769	AB	AB9	Alliance	Warrior	4	5	12	41076	8009	348	\N	1	dps	\N
2770	AB	AB9	Horde	Rogue	5	2	46	63447	19978	555	1	\N	dps	\N
2771	AB	AB9	Horde	Druid	4	2	60	48223	8191	422	1	\N	dps	\N
2772	AB	AB9	Alliance	Demon Hunter	5	4	15	57192	9139	352	\N	1	dps	\N
2773	AB	AB9	Horde	Druid	5	1	43	26753	462	130	1	\N	dps	\N
2774	AB	AB9	Alliance	Paladin	4	3	13	21547	10455	349	\N	1	dps	\N
2775	AB	AB9	Alliance	Druid	1	4	12	28490	278	347	\N	1	dps	\N
2776	AB	AB9	Horde	Priest	7	0	59	72947	14646	429	1	\N	dps	\N
2777	AB	AB9	Alliance	Demon Hunter	2	8	14	67503	6576	346	\N	1	dps	\N
2778	AB	AB9	Alliance	Druid	0	6	14	2853	114000	352	\N	1	heal	\N
2779	AB	AB9	Horde	Shaman	2	1	66	14098	103000	589	1	\N	heal	\N
2780	AB	AB9	Alliance	Priest	1	6	15	23349	75091	360	\N	1	heal	\N
2781	AB	AB9	Horde	Priest	4	1	64	16093	81297	287	1	\N	heal	\N
2782	AB	AB9	Alliance	Paladin	0	6	10	51499	20862	347	\N	1	dps	\N
2783	AB	AB9	Horde	Monk	8	3	59	53737	20958	559	1	\N	dps	\N
2784	AB	AB9	Horde	Demon Hunter	1	1	29	29317	6679	383	1	\N	dps	\N
2785	AB	AB9	Horde	Hunter	3	4	46	33327	12025	541	1	\N	dps	\N
2786	AB	AB9	Alliance	Mage	1	4	10	34679	11701	332	\N	1	dps	\N
2787	AB	AB9	Alliance	Hunter	2	9	13	11150	129	341	\N	1	dps	\N
2788	AB	AB9	Alliance	Warrior	4	1	12	35884	12652	358	\N	1	dps	\N
2789	AB	AB9	Horde	Rogue	9	1	56	58301	11934	397	1	\N	dps	\N
2790	AB	AB9	Horde	Warlock	7	4	57	57060	22010	430	1	\N	dps	\N
2791	AB	AB9	Horde	Rogue	5	5	41	69940	17647	236	1	\N	dps	\N
2792	AB	AB9	Alliance	Hunter	0	9	18	26934	15295	359	\N	1	dps	\N
2793	AB	AB9	Horde	Shaman	10	4	59	75840	11148	566	1	\N	dps	\N
2794	AB	AB9	Alliance	Hunter	0	8	13	33535	8935	347	\N	1	dps	\N
2795	AB	AB9	Alliance	Warrior	2	9	17	36034	3591	354	\N	1	dps	\N
2796	AB	AB9	Horde	Druid	11	0	74	57527	6663	621	1	\N	dps	\N
2797	AB	AB9	Alliance	Rogue	3	4	20	20672	13754	372	\N	1	dps	\N
2798	AB	AB10	Alliance	Mage	1	5	17	44944	11552	239	\N	1	dps	\N
2799	AB	AB10	Alliance	Shaman	0	8	18	3538	50332	241	\N	1	heal	\N
2800	AB	AB10	Horde	Rogue	1	0	28	41642	5374	387	1	\N	dps	\N
2801	AB	AB10	Horde	Priest	11	2	48	79277	14164	544	1	\N	dps	\N
2802	AB	AB10	Alliance	Warrior	1	6	16	54428	7977	238	\N	1	dps	\N
2803	AB	AB10	Horde	Monk	2	1	40	26218	15029	388	1	\N	dps	\N
2804	AB	AB10	Horde	Mage	8	1	31	60571	1735	523	1	\N	dps	\N
2805	AB	AB10	Alliance	Rogue	1	4	19	19669	10734	246	\N	1	dps	\N
2806	AB	AB10	Alliance	Warrior	2	4	19	28076	4369	244	\N	1	dps	\N
2807	AB	AB10	Horde	Death Knight	2	1	34	24614	53293	382	1	\N	dps	\N
2808	AB	AB10	Alliance	Paladin	3	5	19	118000	28467	253	\N	1	dps	\N
2809	AB	AB10	Horde	Priest	1	2	43	29031	83687	394	1	\N	heal	\N
2810	AB	AB10	Horde	Shaman	0	3	48	7891	89729	395	1	\N	heal	\N
2811	AB	AB10	Alliance	Warlock	4	5	17	53285	27152	239	\N	1	dps	\N
2812	AB	AB10	Horde	Shaman	0	1	43	7074	121000	393	1	\N	heal	\N
2813	AB	AB10	Horde	Paladin	4	3	39	57822	24251	385	1	\N	dps	\N
2814	AB	AB10	Alliance	Hunter	2	3	21	31647	6484	254	\N	1	dps	\N
2815	AB	AB10	Horde	Druid	3	0	32	39659	5570	376	1	\N	dps	\N
2816	AB	AB10	Alliance	Mage	1	3	20	58149	15114	251	\N	1	dps	\N
2817	AB	AB10	Alliance	Druid	1	0	5	4476	0	153	\N	1	dps	\N
2818	AB	AB10	Alliance	Mage	1	3	6	26645	8720	218	\N	1	dps	\N
2819	AB	AB10	Horde	Mage	4	3	11	22335	5918	365	1	\N	dps	\N
2820	AB	AB10	Alliance	Mage	1	2	18	28407	20512	242	\N	1	dps	\N
2821	AB	AB10	Alliance	Shaman	0	6	11	10477	74377	228	\N	1	heal	\N
2822	AB	AB10	Horde	Warrior	4	3	46	38283	6295	395	1	\N	dps	\N
2823	AB	AB10	Horde	Priest	10	1	47	94373	18876	391	1	\N	dps	\N
2824	AB	AB10	Alliance	Rogue	2	1	15	14029	1934	239	\N	1	dps	\N
2825	AB	AB10	Alliance	Warrior	0	1	1	4967	1323	145	\N	1	dps	\N
2826	AB	AB10	Horde	Warrior	5	3	15	47579	8758	358	1	\N	dps	\N
2827	AB	AB10	Horde	Shaman	4	0	42	37471	6675	383	1	\N	dps	\N
2828	BG	BG28	Horde	Rogue	1	3	19	12815	8315	226	\N	1	dps	\N
2829	BG	BG28	Alliance	Mage	0	3	38	32244	5841	829	1	\N	dps	\N
2830	BG	BG28	Horde	Druid	5	8	19	51777	1491	233	\N	1	dps	\N
2831	BG	BG28	Horde	Warlock	5	4	21	52035	25280	238	\N	1	dps	\N
2832	BG	BG28	Alliance	Paladin	9	2	47	81704	14283	867	1	\N	dps	\N
2833	BG	BG28	Horde	Rogue	1	8	16	26180	17830	218	\N	1	dps	\N
2834	BG	BG28	Horde	Shaman	0	11	15	11146	69345	219	\N	1	heal	\N
2835	BG	BG28	Alliance	Druid	15	3	44	85682	10930	849	1	\N	dps	\N
2836	BG	BG28	Alliance	Paladin	1	2	15	16469	12671	538	1	\N	dps	\N
2837	BG	BG28	Alliance	Demon Hunter	10	3	45	82541	16073	861	1	\N	dps	\N
2838	BG	BG28	Horde	Warlock	2	12	16	50784	30645	220	\N	1	dps	\N
2839	BG	BG28	Alliance	Mage	8	4	37	63112	6946	832	1	\N	dps	\N
2840	BG	BG28	Horde	Warlock	7	6	23	68785	44256	237	\N	1	dps	\N
2841	BG	BG28	Alliance	Monk	2	3	41	8886	118000	839	1	\N	heal	\N
2842	BG	BG28	Alliance	Warlock	7	3	41	62628	24025	844	1	\N	dps	\N
2843	BG	BG28	Horde	Warlock	1	1	14	6715	8909	217	\N	1	dps	\N
2844	BG	BG28	Horde	Druid	0	2	16	20666	16116	218	\N	1	dps	\N
2845	BG	BG28	Alliance	Warrior	5	3	39	50327	4656	832	1	\N	dps	\N
2846	BG	BG28	Alliance	Rogue	4	5	38	61225	9215	608	1	\N	dps	\N
2847	BG	BG28	Horde	Paladin	8	6	27	87013	13419	250	\N	1	dps	\N
2848	BG	BG29	Horde	Priest	7	4	23	58117	27284	516	1	\N	dps	\N
2849	BG	BG29	Horde	Paladin	1	0	14	7888	8930	500	1	\N	heal	\N
2850	BG	BG29	Alliance	Priest	3	5	13	30613	12804	202	\N	1	dps	\N
2851	BG	BG29	Alliance	Mage	1	1	12	16025	3195	198	\N	1	dps	\N
2852	BG	BG29	Horde	Paladin	0	1	22	1739	18596	514	1	\N	heal	\N
2853	BG	BG29	Horde	Druid	1	3	19	21357	3923	363	1	\N	dps	\N
2854	BG	BG29	Horde	Shaman	0	1	25	8958	60665	522	1	\N	heal	\N
2855	BG	BG29	Alliance	Druid	0	1	11	4022	32412	194	\N	1	heal	\N
2856	BG	BG29	Alliance	Paladin	3	2	13	40414	15552	205	\N	1	dps	\N
2857	BG	BG29	Alliance	Hunter	0	0	12	17326	93	198	\N	1	dps	\N
2858	BG	BG29	Alliance	Warlock	0	5	13	17274	3538	202	\N	1	dps	\N
2859	BG	BG29	Horde	Rogue	4	1	23	30965	8881	520	1	\N	dps	\N
2860	BG	BG29	Horde	Warlock	9	1	25	67340	14639	375	1	\N	dps	\N
2861	BG	BG29	Alliance	Druid	0	4	13	1131	70329	205	\N	1	heal	\N
2862	BG	BG29	Alliance	Demon Hunter	4	4	15	27774	2857	211	\N	1	dps	\N
2863	BG	BG29	Horde	Rogue	2	2	12	28297	2544	498	1	\N	dps	\N
2864	BG	BG29	Horde	Warlock	6	1	25	56087	22772	525	1	\N	dps	\N
2865	BG	BG29	Horde	Demon Hunter	2	1	25	46473	6132	521	1	\N	dps	\N
2866	BG	BG29	Alliance	Shaman	3	3	15	57402	3008	211	\N	1	dps	\N
2867	BG	BG29	Alliance	Warrior	1	6	12	28500	4106	198	\N	1	dps	\N
2868	BG	BG30	Horde	Mage	3	2	31	39138	7526	388	1	\N	dps	\N
2869	BG	BG30	Horde	Warrior	2	0	5	4619	440	380	1	\N	dps	\N
2870	BG	BG30	Alliance	Warlock	7	4	24	42620	20739	275	\N	1	dps	\N
2871	BG	BG30	Alliance	Rogue	2	6	16	21629	14824	246	\N	1	dps	\N
2872	BG	BG30	Horde	Warrior	7	6	25	67785	9924	539	1	\N	dps	\N
2873	BG	BG30	Alliance	Mage	5	2	23	50713	9909	272	\N	1	dps	\N
2874	BG	BG30	Alliance	Shaman	2	4	17	28973	6138	239	\N	1	dps	\N
2875	BG	BG30	Horde	Shaman	7	5	30	65704	16471	553	1	\N	dps	\N
2876	BG	BG30	Horde	Warrior	15	2	35	80974	19324	404	1	\N	dps	\N
2877	BG	BG30	Horde	Rogue	6	2	37	34201	4741	561	1	\N	dps	\N
2878	BG	BG30	Horde	Shaman	0	2	15	3634	50517	506	1	\N	heal	\N
2879	BG	BG30	Alliance	Hunter	2	5	17	25822	6597	251	\N	1	dps	\N
2880	BG	BG30	Alliance	Death Knight	1	0	6	7090	1774	161	\N	1	dps	\N
2881	BG	BG30	Alliance	Priest	0	3	6	3862	8559	188	\N	1	heal	\N
2882	BG	BG30	Alliance	Warlock	4	7	15	41154	23171	245	\N	1	dps	\N
2883	BG	BG30	Alliance	Monk	1	4	17	33703	29858	251	\N	1	dps	\N
2884	BG	BG30	Horde	Monk	0	1	7	2250	18789	485	1	\N	heal	\N
2885	BG	BG30	Horde	Death Knight	3	5	32	84194	34559	555	1	\N	dps	\N
2886	BG	BG30	Alliance	Mage	0	5	19	20289	0	257	\N	1	dps	\N
2887	BG	BG31	Alliance	Druid	4	9	31	42704	1675	469	\N	1	dps	1
2888	BG	BG31	Horde	Mage	6	7	46	95644	10628	490	1	\N	dps	1
2889	BG	BG31	Alliance	Warlock	3	8	23	44196	59369	434	\N	1	dps	1
2890	BG	BG31	Alliance	Druid	0	8	36	22639	112000	476	\N	1	heal	1
2891	BG	BG31	Alliance	Warlock	2	2	33	48988	22759	463	\N	1	dps	1
2892	BG	BG31	Alliance	Paladin	11	5	35	140000	39553	487	\N	1	dps	1
2893	BG	BG31	Horde	Rogue	10	4	42	90032	22282	713	1	\N	dps	1
2894	BG	BG31	Alliance	Priest	0	3	25	11637	73700	448	\N	1	heal	1
2895	BG	BG31	Horde	Shaman	0	10	32	19747	136000	462	1	\N	heal	1
2896	BG	BG31	Alliance	Mage	0	3	18	22910	11304	415	\N	1	dps	1
2897	BG	BG31	Horde	Paladin	4	4	13	52595	11601	653	1	\N	dps	1
2898	BG	BG31	Alliance	Druid	5	5	19	69221	38254	424	\N	1	dps	1
2899	BG	BG31	Alliance	Priest	16	7	42	148000	55199	501	\N	1	dps	1
2900	BG	BG31	Horde	Death Knight	1	6	44	80907	44322	488	1	\N	dps	1
2901	BG	BG31	Horde	Warrior	14	5	48	79489	15892	725	1	\N	dps	1
2902	BG	BG31	Alliance	Warlock	11	9	29	80798	50254	476	\N	1	dps	1
2903	BG	BG31	Horde	Rogue	0	4	40	67392	20763	705	1	\N	dps	1
2904	BG	BG31	Horde	Death Knight	8	5	46	97246	51520	493	1	\N	dps	1
2905	BG	BG31	Horde	Rogue	4	3	36	53968	15927	701	1	\N	dps	1
2906	BG	BG31	Horde	Priest	12	5	48	134000	42916	721	1	\N	dps	1
2907	BG	BG32	Alliance	Hunter	0	5	1	19000	539	161	\N	1	dps	\N
2908	BG	BG32	Alliance	Druid	0	2	0	14087	279	30	\N	1	dps	\N
2909	BG	BG32	Horde	Mage	0	0	9	12033	75	338	1	\N	dps	\N
2910	BG	BG32	Horde	Warrior	0	0	9	7010	0	488	1	\N	dps	\N
2911	BG	BG32	Horde	Warrior	10	0	25	36194	2949	371	1	\N	dps	\N
2912	BG	BG32	Horde	Mage	2	1	18	17669	2614	506	1	\N	dps	\N
2913	BG	BG32	Alliance	Priest	0	1	0	0	13800	157	\N	1	heal	\N
2914	BG	BG32	Horde	Demon Hunter	2	0	26	28866	2111	524	1	\N	dps	\N
2915	BG	BG32	Horde	Shaman	1	0	20	2402	37528	513	1	\N	heal	\N
2916	BG	BG32	Alliance	Priest	0	4	0	19784	4162	157	\N	1	dps	\N
2917	BG	BG32	Alliance	Mage	0	1	0	15014	3587	142	\N	1	dps	\N
2918	BG	BG32	Horde	Rogue	6	0	21	39651	1908	364	1	\N	dps	\N
2919	BG	BG32	Horde	Mage	0	0	11	11127	7799	345	1	\N	dps	\N
2920	BG	BG32	Alliance	Rogue	0	2	0	16016	3347	157	\N	1	dps	\N
2921	BG	BG32	Alliance	Paladin	1	4	1	6135	43753	161	\N	1	heal	\N
2922	BG	BG32	Alliance	Hunter	0	3	1	17262	4017	161	\N	1	dps	\N
2923	BG	BG32	Horde	Shaman	0	0	21	5429	52563	364	1	\N	heal	\N
2924	BG	BG32	Alliance	Mage	0	4	1	13920	5276	161	\N	1	dps	\N
2925	BG	BG32	Horde	Paladin	6	0	25	57045	5153	368	1	\N	dps	\N
2926	DG	DG3	Alliance	Warlock	0	5	27	76008	25701	330	\N	1	dps	\N
2927	DG	DG3	Alliance	Demon Hunter	3	2	30	144000	8315	333	\N	1	dps	\N
2928	DG	DG3	Horde	Warlock	3	3	38	116000	49956	391	1	\N	dps	\N
2929	DG	DG3	Horde	Warrior	4	3	38	117000	21180	403	1	\N	dps	\N
2930	DG	DG3	Horde	Rogue	4	0	32	84237	8528	533	1	\N	dps	\N
2931	DG	DG3	Alliance	Shaman	0	3	32	5696	205000	344	\N	1	heal	\N
2932	DG	DG3	Alliance	Druid	1	2	29	2841	163000	337	\N	1	heal	\N
2933	DG	DG3	Horde	Priest	4	2	41	131000	20761	550	1	\N	dps	\N
2934	DG	DG3	Alliance	Shaman	0	4	31	9521	215000	337	\N	1	heal	\N
2935	DG	DG3	Horde	Mage	10	1	44	99434	8214	562	1	\N	dps	\N
2936	DG	DG3	Alliance	Druid	4	3	31	132000	22714	340	\N	1	dps	\N
2937	DG	DG3	Horde	Shaman	0	3	39	16967	133000	403	1	\N	heal	\N
2938	DG	DG3	Alliance	Paladin	1	2	31	54369	44113	337	\N	1	dps	\N
2939	DG	DG3	Horde	Paladin	2	0	14	23203	10270	258	1	\N	dps	\N
2940	DG	DG3	Alliance	Druid	2	6	14	84957	6825	299	\N	1	dps	\N
2941	DG	DG3	Alliance	Death Knight	4	6	20	93363	20512	313	\N	1	dps	\N
2942	DG	DG3	Alliance	Mage	0	5	23	53853	1229	323	\N	1	dps	\N
2943	DG	DG3	Alliance	Mage	4	2	32	135000	17703	351	\N	1	dps	\N
2944	DG	DG3	Horde	Monk	2	3	38	68113	30423	550	1	\N	dps	\N
2945	DG	DG3	Alliance	Death Knight	3	3	23	65827	27404	330	\N	1	dps	\N
2946	DG	DG3	Horde	Paladin	0	4	40	125000	39998	404	1	\N	dps	\N
2947	DG	DG3	Horde	Mage	8	1	41	155000	17595	404	1	\N	dps	\N
2948	DG	DG3	Alliance	Druid	1	2	30	111000	28565	333	\N	1	dps	\N
2949	DG	DG3	Horde	Druid	0	0	36	3192	170000	396	1	\N	heal	\N
2950	DG	DG3	Horde	Mage	1	3	35	42831	16161	395	1	\N	dps	\N
2951	DG	DG3	Alliance	Mage	9	1	31	134000	10375	338	\N	1	dps	\N
2952	DG	DG3	Alliance	Druid	0	2	30	2714	185000	338	\N	1	heal	\N
2953	DG	DG3	Horde	Warrior	5	5	26	76800	15195	380	1	\N	dps	\N
2954	DG	DG3	Horde	Paladin	5	3	38	149000	50323	397	1	\N	dps	\N
2955	DG	DG3	Horde	Druid	0	2	41	2359	325000	400	1	\N	heal	\N
2956	ES	ES8	Alliance	Priest	0	1	12	9355	40727	316	\N	1	heal	\N
2957	ES	ES8	Alliance	Paladin	2	4	22	50832	8713	420	\N	1	dps	\N
2958	ES	ES8	Alliance	Demon Hunter	3	4	26	138000	9792	424	\N	1	dps	\N
2959	ES	ES8	Horde	Warrior	0	3	41	80202	48995	465	1	\N	dps	\N
2960	ES	ES8	Horde	Mage	3	0	42	188000	9493	466	1	\N	dps	\N
2961	ES	ES8	Alliance	Monk	0	0	13	1063	160000	347	\N	1	heal	\N
2962	ES	ES8	Alliance	Priest	2	4	15	5551	137000	383	\N	1	heal	\N
2963	ES	ES8	Horde	Paladin	1	1	44	6011	185000	470	1	\N	heal	\N
2964	ES	ES8	Alliance	Druid	2	6	23	108000	3039	412	\N	1	dps	\N
2965	ES	ES8	Horde	Druid	0	5	30	43864	7315	453	1	\N	dps	\N
2966	ES	ES8	Horde	Warrior	7	3	36	112000	25158	459	1	\N	dps	\N
2967	ES	ES8	Horde	Shaman	0	3	41	23211	142000	621	1	\N	heal	\N
2968	ES	ES8	Horde	Monk	2	1	42	129000	46042	465	1	\N	dps	\N
2969	ES	ES8	Horde	Warrior	2	6	37	53393	12905	459	1	\N	dps	\N
2970	ES	ES8	Horde	Rogue	1	2	40	63907	13285	615	1	\N	dps	\N
2971	ES	ES8	Alliance	Druid	0	3	24	94980	56758	411	\N	1	dps	\N
2972	ES	ES8	Alliance	Mage	3	2	25	224000	22373	413	\N	1	dps	\N
2973	ES	ES8	Alliance	Monk	2	6	23	61271	30811	412	\N	1	dps	\N
2974	ES	ES8	Alliance	Paladin	0	4	23	10883	194000	408	\N	1	heal	\N
2975	ES	ES8	Alliance	Monk	0	1	9	16071	10070	380	\N	1	dps	\N
2976	ES	ES8	Alliance	Priest	0	1	28	962	186000	422	\N	1	heal	\N
2977	ES	ES8	Horde	Rogue	7	4	39	69127	7712	466	1	\N	dps	\N
2978	ES	ES8	Alliance	Priest	5	3	26	99575	23554	416	\N	1	dps	\N
2979	ES	ES8	Horde	Druid	7	1	42	95022	39650	469	1	\N	dps	\N
2980	ES	ES8	Horde	Mage	8	0	43	227000	14281	466	1	\N	dps	\N
2981	ES	ES8	Alliance	Hunter	8	2	27	85223	1140	434	\N	1	dps	\N
2982	ES	ES8	Alliance	Paladin	0	3	24	5058	146000	411	\N	1	heal	\N
2983	ES	ES8	Horde	Druid	0	0	45	433	304000	468	1	\N	heal	\N
2984	ES	ES8	Horde	Demon Hunter	4	2	35	81648	9546	463	1	\N	dps	\N
2985	ES	ES8	Horde	Death Knight	1	1	38	86780	15694	465	1	\N	dps	\N
2986	ES	ES9	Alliance	Shaman	0	3	25	8382	63387	460	\N	1	heal	\N
2987	ES	ES9	Alliance	Demon Hunter	2	6	26	45233	14510	464	\N	1	dps	\N
2988	ES	ES9	Alliance	Shaman	2	10	21	66603	23836	449	\N	1	dps	\N
2989	ES	ES9	Horde	Priest	2	3	70	27087	130000	645	1	\N	heal	\N
2990	ES	ES9	Alliance	Warrior	0	9	31	30814	19313	466	\N	1	dps	\N
2991	ES	ES9	Alliance	Priest	4	5	25	59249	27552	458	\N	1	dps	\N
2992	ES	ES9	Alliance	Hunter	0	4	17	23065	11465	435	\N	1	dps	\N
2993	ES	ES9	Alliance	Druid	3	3	34	79261	18919	480	\N	1	dps	\N
2994	ES	ES9	Horde	Rogue	1	2	66	36043	6627	634	1	\N	dps	\N
2995	ES	ES9	Alliance	Paladin	1	9	24	50849	20390	448	\N	1	dps	\N
2996	ES	ES9	Horde	Priest	5	1	65	45180	124000	491	1	\N	heal	\N
2997	ES	ES9	Horde	Priest	17	0	71	97789	23242	650	1	\N	dps	\N
2998	ES	ES9	Alliance	Warlock	2	6	29	76313	38347	463	\N	1	dps	\N
2999	ES	ES9	Alliance	Demon Hunter	6	2	33	83263	16703	480	\N	1	dps	\N
3000	ES	ES9	Horde	Shaman	2	4	60	25150	109000	638	1	\N	heal	\N
3001	ES	ES9	Horde	Hunter	5	3	61	47007	4222	624	1	\N	dps	\N
3002	ES	ES9	Horde	Warlock	9	4	53	117000	48282	632	1	\N	dps	\N
3003	ES	ES9	Horde	Rogue	2	4	52	54620	14834	465	1	\N	dps	\N
3004	ES	ES9	Alliance	Warrior	3	6	28	60962	15962	463	\N	1	dps	\N
3005	ES	ES9	Horde	Mage	13	0	76	63305	5593	646	1	\N	dps	\N
3006	ES	ES9	Alliance	Shaman	0	8	27	0	79920	464	\N	1	heal	\N
3007	ES	ES9	Alliance	Paladin	4	4	28	107000	26645	473	\N	1	dps	\N
3008	ES	ES9	Horde	Priest	1	2	73	12763	80446	487	1	\N	heal	\N
3009	ES	ES9	Horde	Death Knight	2	3	51	61486	19281	462	1	\N	dps	\N
3010	ES	ES9	Horde	Paladin	12	4	54	87713	34271	607	1	\N	dps	\N
3011	ES	ES9	Horde	Demon Hunter	5	3	64	82171	11230	620	1	\N	dps	\N
3012	ES	ES9	Horde	Rogue	3	2	54	40159	13474	625	1	\N	dps	\N
3013	ES	ES9	Alliance	Hunter	5	6	26	64653	14764	463	\N	1	dps	\N
3014	ES	ES9	Alliance	Demon Hunter	4	3	21	44947	2643	399	\N	1	dps	\N
3015	ES	ES9	Horde	Warrior	5	4	54	69011	16652	478	1	\N	dps	\N
3016	ES	ES10	Alliance	Shaman	2	3	13	18617	10443	315	\N	1	dps	\N
3017	ES	ES10	Horde	Druid	0	2	32	1146	1687	511	1	\N	heal	\N
3018	ES	ES10	Alliance	Demon Hunter	4	5	18	95768	14535	338	\N	1	dps	\N
3019	ES	ES10	Alliance	Warrior	3	6	18	64353	13098	337	\N	1	dps	\N
3020	ES	ES10	Horde	Hunter	2	3	35	66666	11708	672	1	\N	dps	\N
3021	ES	ES10	Horde	Death Knight	5	2	36	59321	11405	676	1	\N	dps	\N
3022	ES	ES10	Alliance	Druid	4	3	17	52229	2794	329	\N	1	dps	\N
3023	ES	ES10	Horde	Paladin	2	1	42	18828	4041	680	1	\N	dps	\N
3024	ES	ES10	Horde	Warrior	8	2	34	77016	16975	513	1	\N	dps	\N
3025	ES	ES10	Alliance	Warlock	3	2	16	38533	11527	332	\N	1	dps	\N
3026	ES	ES10	Horde	Shaman	1	2	32	9816	87543	670	1	\N	heal	\N
3027	ES	ES10	Horde	Druid	0	2	37	11217	130000	675	1	\N	heal	\N
3028	ES	ES10	Horde	Monk	1	4	29	24588	20117	670	1	\N	dps	\N
3029	ES	ES10	Alliance	Rogue	3	1	20	62068	11835	339	\N	1	dps	\N
3030	ES	ES10	Alliance	Warrior	0	11	9	30534	9282	307	\N	1	dps	\N
3031	ES	ES10	Alliance	Hunter	0	1	1	10231	1544	145	\N	1	dps	\N
3032	ES	ES10	Horde	Demon Hunter	6	2	44	58118	2840	671	1	\N	dps	\N
3033	ES	ES10	Alliance	Paladin	1	3	14	92768	25234	321	\N	1	dps	\N
3034	ES	ES10	Horde	Paladin	0	2	27	2219	54562	508	1	\N	heal	\N
3035	ES	ES10	Alliance	Druid	5	3	19	29420	13115	350	\N	1	dps	\N
3036	ES	ES10	Horde	Monk	4	3	31	60937	34725	519	1	\N	dps	\N
3037	ES	ES10	Alliance	Shaman	0	5	13	1359	63234	322	\N	1	heal	\N
3038	ES	ES10	Horde	Paladin	5	3	37	52120	22092	665	1	\N	dps	\N
3039	ES	ES10	Horde	Mage	5	0	44	80580	7330	688	1	\N	dps	\N
3040	ES	ES10	Alliance	Druid	0	1	11	16675	4795	318	\N	1	dps	\N
3041	ES	ES10	Horde	Shaman	4	1	46	25377	9015	679	1	\N	dps	\N
3042	ES	ES10	Horde	Warlock	7	1	36	110000	35275	670	1	\N	dps	\N
3043	ES	ES10	Alliance	Rogue	1	6	13	38795	12603	322	\N	1	dps	\N
3044	ES	ES10	Alliance	Rogue	3	3	15	33474	10036	190	\N	1	dps	\N
3045	ES	ES10	Alliance	Priest	0	3	21	13184	155000	345	\N	1	heal	\N
3046	ES	ES11	Horde	Warlock	5	0	46	31604	10939	657	1	\N	dps	\N
3047	ES	ES11	Alliance	Druid	2	8	17	36320	11188	275	\N	1	dps	\N
3048	ES	ES11	Horde	Paladin	4	2	46	58400	27829	515	1	\N	dps	\N
3049	ES	ES11	Horde	Hunter	3	7	37	26175	7928	490	1	\N	dps	\N
3050	ES	ES11	Alliance	Hunter	1	1	6	29278	3931	170	\N	1	dps	\N
3051	ES	ES11	Horde	Monk	0	1	60	74	133000	662	1	\N	heal	\N
3052	ES	ES11	Alliance	Monk	0	3	24	4504	150000	292	\N	1	heal	\N
3053	ES	ES11	Alliance	Demon Hunter	0	6	21	34807	6321	279	\N	1	dps	\N
3054	ES	ES11	Horde	Demon Hunter	2	4	57	54784	18487	511	1	\N	dps	\N
3055	ES	ES11	Horde	Priest	7	3	51	97631	29796	660	1	\N	dps	\N
3056	ES	ES11	Alliance	Rogue	2	0	23	23656	11028	291	\N	1	dps	\N
3057	ES	ES11	Horde	Mage	3	2	55	77421	23633	506	1	\N	dps	\N
3058	ES	ES11	Alliance	Druid	3	6	24	61061	4465	291	\N	1	dps	\N
3059	ES	ES11	Horde	Shaman	2	2	62	9468	97041	687	1	\N	heal	\N
3060	ES	ES11	Alliance	Rogue	2	6	18	59202	4178	278	\N	1	dps	\N
3061	ES	ES11	Horde	Druid	2	2	49	42140	25053	461	1	\N	dps	\N
3062	ES	ES11	Horde	Priest	2	1	58	20859	162000	673	1	\N	heal	\N
3063	ES	ES11	Alliance	Death Knight	2	6	17	38045	86757	256	\N	1	dps	\N
3064	ES	ES11	Horde	Demon Hunter	4	3	58	60963	8505	521	1	\N	dps	\N
3065	ES	ES11	Horde	Death Knight	7	2	57	63330	25606	527	1	\N	dps	\N
3066	ES	ES11	Alliance	Shaman	6	5	24	79813	11975	288	\N	1	dps	\N
3067	ES	ES11	Alliance	Demon Hunter	0	3	24	53385	9603	284	\N	1	dps	\N
3068	ES	ES11	Alliance	Shaman	4	5	21	54133	16587	287	\N	1	dps	\N
3069	ES	ES11	Alliance	Death Knight	3	6	17	60940	23735	271	\N	1	dps	\N
3070	ES	ES11	Alliance	Shaman	3	4	26	111000	7504	304	\N	1	dps	\N
3071	ES	ES11	Horde	Priest	20	1	65	110000	259095	528	1	\N	heal	\N
3072	ES	ES11	Alliance	Mage	0	8	17	32773	6079	275	\N	1	dps	\N
3073	ES	ES11	Alliance	Mage	1	3	25	64138	12328	289	\N	1	dps	\N
3074	ES	ES11	Horde	Warlock	3	2	34	41211	23723	498	1	\N	dps	\N
3075	ES	ES11	Horde	Warlock	7	3	51	99156	48362	506	1	\N	dps	\N
3076	ES	ES12	Alliance	Monk	0	4	61	2022	212000	987	1	\N	heal	\N
3077	ES	ES12	Horde	Mage	5	6	26	73649	24276	324	\N	1	dps	\N
3078	ES	ES12	Horde	Priest	2	3	22	46930	16874	235	\N	1	dps	\N
3079	ES	ES12	Alliance	Hunter	4	0	38	37528	2338	534	1	\N	dps	\N
3080	ES	ES12	Alliance	Shaman	2	3	65	42002	9194	762	1	\N	dps	\N
3081	ES	ES12	Horde	Death Knight	2	7	30	74343	47206	318	\N	1	dps	\N
3082	ES	ES12	Alliance	Shaman	12	2	62	101000	15086	742	1	\N	dps	\N
3083	ES	ES12	Horde	Shaman	1	5	28	20798	110000	319	\N	1	heal	\N
3084	ES	ES12	Horde	Demon Hunter	4	6	29	29498	3545	312	\N	1	dps	\N
3085	ES	ES12	Horde	Death Knight	5	8	24	150000	40917	300	\N	1	dps	\N
3086	ES	ES12	Horde	Death Knight	4	6	25	69872	24057	311	\N	1	dps	\N
3087	ES	ES12	Alliance	Shaman	8	0	69	99195	11009	996	1	\N	dps	\N
3088	ES	ES12	Horde	Mage	1	6	25	69169	34883	308	\N	1	dps	\N
3089	ES	ES12	Horde	Druid	0	5	26	8471	126000	308	\N	1	heal	\N
3090	ES	ES12	Alliance	Mage	2	3	56	16771	8668	972	1	\N	dps	\N
3091	ES	ES12	Horde	Rogue	1	2	33	40319	16479	326	\N	1	dps	\N
3092	ES	ES12	Horde	Priest	0	7	31	11922	49871	318	\N	1	heal	\N
3093	ES	ES12	Alliance	Death Knight	3	1	54	97249	29479	967	1	\N	dps	\N
3094	ES	ES12	Alliance	Death Knight	7	4	65	116000	38550	760	1	\N	dps	\N
3095	ES	ES12	Horde	Hunter	0	9	20	50272	16224	299	\N	1	dps	\N
3096	ES	ES12	Alliance	Shaman	2	9	38	52859	15653	704	1	\N	dps	\N
3097	ES	ES12	Alliance	Druid	8	2	52	83112	6371	973	1	\N	dps	\N
3098	ES	ES12	Horde	Warrior	5	5	27	70136	11294	313	\N	1	dps	\N
3099	ES	ES12	Alliance	Monk	0	4	57	5686	220000	975	1	\N	heal	\N
3100	ES	ES12	Alliance	Druid	10	4	64	147000	1981	996	1	\N	dps	\N
3101	ES	ES12	Alliance	Warlock	3	3	53	42685	33000	972	1	\N	dps	\N
3102	ES	ES12	Horde	Warlock	4	2	32	40892	16298	320	\N	1	dps	\N
3103	ES	ES12	Alliance	Warrior	1	1	34	22080	8130	744	1	\N	dps	\N
3104	ES	ES12	Alliance	Rogue	6	3	49	40456	8674	956	1	\N	dps	\N
3105	ES	ES12	Horde	Druid	6	5	25	156000	8610	320	\N	1	dps	\N
3106	ES	ES13	Horde	Hunter	6	0	44	48413	7272	551	1	\N	dps	\N
3107	ES	ES13	Horde	Paladin	1	0	43	3435	84500	552	1	\N	heal	\N
3108	ES	ES13	Horde	Shaman	0	0	36	5009	44391	394	1	\N	heal	\N
3109	ES	ES13	Horde	Warlock	6	1	42	54633	13479	398	1	\N	dps	\N
3110	ES	ES13	Alliance	Warrior	1	4	6	29807	1918	171	\N	1	dps	\N
3111	ES	ES13	Alliance	Druid	0	0	9	1402	66133	175	\N	1	heal	\N
3112	ES	ES13	Alliance	Warrior	1	1	1	16842	16767	150	\N	1	dps	\N
3113	ES	ES13	Horde	Death Knight	5	1	41	63125	10425	549	1	\N	dps	\N
3114	ES	ES13	Alliance	Shaman	1	4	8	11375	34592	174	\N	1	heal	\N
3115	ES	ES13	Alliance	Priest	0	2	7	3270	35877	172	\N	1	heal	\N
3116	ES	ES13	Horde	Hunter	2	1	41	53289	4734	396	1	\N	dps	\N
3117	ES	ES13	Alliance	Warrior	0	3	4	12232	2965	165	\N	1	dps	\N
3118	ES	ES13	Horde	Warlock	3	1	35	27858	9393	555	1	\N	dps	\N
3119	ES	ES13	Horde	Warlock	3	1	31	19806	6880	398	1	\N	dps	\N
3120	ES	ES13	Horde	Monk	4	1	43	42035	9953	397	1	\N	dps	\N
3121	ES	ES13	Horde	Warrior	2	2	33	19510	4488	391	1	\N	dps	\N
3122	ES	ES13	Horde	Death Knight	1	0	31	37043	11985	537	1	\N	dps	\N
3123	ES	ES13	Alliance	Priest	0	4	7	428	60033	171	\N	1	heal	\N
3124	ES	ES13	Horde	Mage	7	0	47	51246	4447	556	1	\N	dps	\N
3125	ES	ES13	Alliance	Rogue	1	2	8	9090	3572	174	\N	1	dps	\N
3126	ES	ES13	Alliance	Warlock	0	3	7	7012	7803	172	\N	1	dps	\N
3127	ES	ES13	Alliance	Mage	2	3	8	27418	5519	174	\N	1	dps	\N
3128	ES	ES13	Alliance	Warrior	0	5	7	28573	2018	50	\N	1	dps	\N
3129	ES	ES13	Horde	Death Knight	4	0	38	46029	3950	399	1	\N	dps	\N
3130	ES	ES13	Alliance	Druid	0	1	7	750	3480	29	\N	1	heal	\N
3131	ES	ES13	Alliance	Warrior	3	6	8	36268	4875	175	\N	1	dps	\N
3132	ES	ES13	Alliance	Hunter	0	4	5	22737	4152	167	\N	1	dps	\N
3133	ES	ES13	Horde	Rogue	1	0	39	17363	1673	396	1	\N	dps	\N
3134	ES	ES13	Alliance	Death Knight	1	3	9	57050	13802	181	\N	1	dps	\N
3135	ES	ES13	Horde	Demon Hunter	0	2	41	6613	0	547	1	\N	dps	\N
3136	SA	SA3	Horde	Rogue	6	9	59	88344	14586	535	1	\N	dps	\N
3137	SA	SA3	Alliance	Druid	6	8	40	176000	28455	341	\N	1	dps	\N
3138	SA	SA3	Alliance	Mage	3	8	48	83221	18905	357	\N	1	dps	\N
3139	SA	SA3	Alliance	Priest	0	8	50	6062	256000	349	\N	1	heal	\N
3140	SA	SA3	Horde	Priest	3	3	95	37231	194000	568	1	\N	heal	\N
3141	SA	SA3	Horde	Paladin	1	3	92	13000	188000	570	1	\N	dps	\N
3142	SA	SA3	Horde	Warrior	0	2	68	36232	50625	528	1	\N	dps	\N
3143	SA	SA3	Horde	Druid	8	5	87	90414	29288	412	1	\N	dps	\N
3144	SA	SA3	Horde	Hunter	11	3	91	90205	3620	558	1	\N	dps	\N
3145	SA	SA3	Horde	Death Knight	2	4	86	43947	46944	386	1	\N	dps	\N
3146	SA	SA3	Alliance	Shaman	0	7	40	12	111000	341	\N	1	heal	\N
3147	SA	SA3	Horde	Rogue	3	3	90	67481	12161	408	1	\N	dps	\N
3148	SA	SA3	Alliance	Shaman	5	6	47	69952	2171	354	\N	1	dps	\N
3149	SA	SA3	Alliance	Demon Hunter	6	7	51	125000	12235	365	\N	1	dps	\N
3150	SA	SA3	Horde	Shaman	9	5	86	129000	21094	552	1	\N	dps	\N
3151	SA	SA3	Horde	Priest	21	3	94	246000	52503	564	1	\N	dps	\N
3152	SA	SA3	Horde	Shaman	2	5	91	38900	142000	416	1	\N	heal	\N
3153	SA	SA3	Alliance	Rogue	0	11	45	73897	3595	351	\N	1	dps	\N
3154	SA	SA3	Alliance	Mage	2	6	40	69771	1189	339	\N	1	dps	\N
3155	SA	SA3	Alliance	Priest	0	8	53	30921	167	368	\N	1	dps	\N
3156	SA	SA3	Horde	Paladin	13	6	92	186000	42491	558	1	\N	dps	\N
3157	SA	SA3	Horde	Death Knight	3	6	87	98052	21409	556	1	\N	dps	\N
3158	SA	SA3	Alliance	Warrior	8	5	51	93865	35817	364	\N	1	dps	\N
3159	SA	SA3	Alliance	Rogue	12	0	54	134000	28637	393	\N	1	dps	\N
3160	SA	SA3	Alliance	Rogue	0	6	47	31331	12833	352	\N	1	dps	\N
3161	SA	SA3	Alliance	Mage	3	4	54	118000	23837	363	\N	1	dps	\N
3162	SA	SA3	Alliance	Demon Hunter	5	5	50	111000	10678	365	\N	1	dps	\N
3163	SA	SA3	Alliance	Death Knight	8	7	54	164000	10233	379	\N	1	dps	\N
3164	SA	SA3	Horde	Druid	3	3	95	63817	51600	568	1	\N	dps	\N
3165	SA	SA3	Horde	Death Knight	9	2	92	124000	23795	553	1	\N	dps	\N
3166	SM	SM20	Alliance	Druid	3	1	33	49910	12993	772	1	\N	dps	\N
3167	SM	SM20	Alliance	Shaman	6	0	36	66867	3966	784	1	\N	dps	\N
3168	SM	SM20	Alliance	Monk	5	0	34	65153	9736	776	1	\N	dps	\N
3169	SM	SM20	Alliance	Mage	1	1	30	30305	3074	770	1	\N	dps	\N
3170	SM	SM20	Horde	Warrior	2	2	6	48017	11851	133	\N	1	dps	\N
3171	SM	SM20	Horde	Warrior	1	0	7	26028	3490	147	\N	1	dps	\N
3172	SM	SM20	Horde	Death Knight	0	3	5	40982	36094	155	\N	1	dps	\N
3173	SM	SM20	Alliance	Demon Hunter	7	0	36	81271	7695	559	1	\N	dps	\N
3174	SM	SM20	Horde	Shaman	0	7	4	8440	70769	153	\N	1	heal	\N
3175	SM	SM20	Horde	Priest	0	5	6	44402	20237	157	\N	1	dps	\N
3176	SM	SM20	Horde	Monk	3	0	6	9627	6494	137	\N	1	dps	\N
3177	SM	SM20	Horde	Demon Hunter	0	4	6	28695	43528	157	\N	1	dps	\N
3178	SM	SM20	Alliance	Priest	0	2	29	18142	33241	540	1	\N	heal	\N
3179	SM	SM20	Horde	Shaman	0	3	4	10593	70359	153	\N	1	heal	\N
3180	SM	SM20	Alliance	Rogue	4	1	34	50209	4896	553	1	\N	dps	\N
3181	SM	SM20	Horde	Hunter	0	5	5	22094	6899	155	\N	1	dps	\N
3182	SM	SM20	Alliance	Warrior	4	0	34	51593	4539	551	1	\N	dps	\N
3183	SM	SM20	Alliance	Priest	0	0	36	3164	133000	784	1	\N	heal	\N
3184	SM	SM20	Alliance	Monk	4	1	34	62042	14284	553	1	\N	dps	\N
3185	SM	SM20	Horde	Warlock	0	5	5	15382	13114	155	\N	1	dps	\N
3186	SM	SM21	Horde	Death Knight	0	5	9	24161	45117	176	\N	1	dps	\N
3187	SM	SM21	Alliance	Hunter	1	1	45	43673	1853	572	1	\N	dps	\N
3188	SM	SM21	Alliance	Priest	2	0	51	25974	49645	809	1	\N	heal	\N
3189	SM	SM21	Horde	Demon Hunter	0	1	1	5062	1771	107	\N	1	dps	\N
3190	SM	SM21	Alliance	Rogue	3	0	51	27467	1919	808	1	\N	dps	\N
3191	SM	SM21	Alliance	Druid	11	1	51	74599	9400	808	1	\N	dps	\N
3192	SM	SM21	Horde	Druid	0	6	9	0	33950	176	\N	1	heal	\N
3193	SM	SM21	Horde	Death Knight	2	3	10	35883	16984	179	\N	1	dps	\N
3194	SM	SM21	Horde	Shaman	0	9	9	4428	70284	177	\N	1	heal	\N
3195	SM	SM21	Horde	Mage	2	3	10	14032	5019	93	\N	1	dps	\N
3196	SM	SM21	Horde	Warlock	0	6	10	74044	44514	178	\N	1	dps	\N
3197	SM	SM21	Horde	Warrior	1	9	9	19075	10962	176	\N	1	dps	\N
3198	SM	SM21	Alliance	Death Knight	2	1	51	62760	21150	583	1	\N	dps	\N
3199	SM	SM21	Alliance	Warrior	7	4	47	79475	14618	802	1	\N	dps	\N
3200	SM	SM21	Horde	Monk	2	2	10	31971	24932	179	\N	1	dps	\N
3201	SM	SM21	Alliance	Priest	5	1	52	65748	10731	812	1	\N	dps	\N
3202	SM	SM21	Alliance	Demon Hunter	13	0	52	81717	6148	812	1	\N	dps	\N
3203	SM	SM21	Horde	Rogue	1	5	2	17185	3208	149	\N	1	dps	\N
3204	SM	SM21	Alliance	Paladin	8	1	51	81430	21647	808	1	\N	dps	\N
3205	SM	SM21	Alliance	Paladin	0	2	51	1527	62209	808	1	\N	heal	\N
3206	SM	SM22	Alliance	Druid	3	2	12	63779	8466	270	\N	1	dps	\N
3207	SM	SM22	Horde	Death Knight	8	0	40	91852	25252	534	1	\N	dps	\N
3208	SM	SM22	Horde	Rogue	1	2	39	13963	6979	532	1	\N	dps	\N
3209	SM	SM22	Horde	Mage	3	1	34	56727	10746	371	1	\N	dps	\N
3210	SM	SM22	Alliance	Druid	0	6	9	35375	12714	258	\N	1	dps	\N
3211	SM	SM22	Alliance	Druid	1	7	9	38562	2969	260	\N	1	dps	\N
3212	SM	SM22	Alliance	Priest	0	5	7	30643	17391	252	\N	1	dps	\N
3213	SM	SM22	Horde	Shaman	7	2	35	77455	12307	524	1	\N	dps	\N
3214	SM	SM22	Horde	Shaman	0	1	39	17303	83105	532	1	\N	heal	\N
3215	SM	SM22	Horde	Hunter	5	2	32	39116	6324	517	1	\N	dps	\N
3216	SM	SM22	Alliance	Druid	0	1	7	1860	26354	253	\N	1	heal	\N
3217	SM	SM22	Alliance	Warrior	1	4	8	21829	8633	247	\N	1	dps	\N
3218	SM	SM22	Horde	Hunter	8	1	37	67151	6595	527	1	\N	dps	\N
3219	SM	SM22	Horde	Priest	2	1	38	11215	82434	527	1	\N	heal	\N
3220	SM	SM22	Horde	Mage	4	1	39	32998	7095	381	1	\N	dps	\N
3221	SM	SM22	Alliance	Shaman	0	3	9	0	80571	260	\N	1	heal	\N
3222	SM	SM22	Horde	Warlock	4	0	39	16625	12874	532	1	\N	dps	\N
3223	SM	SM22	Alliance	Hunter	1	4	10	39089	14840	261	\N	1	dps	\N
3224	SM	SM22	Alliance	Warrior	2	1	6	20931	6434	206	\N	1	dps	\N
3225	SM	SM22	Alliance	Mage	2	6	8	45148	5685	254	\N	1	dps	\N
3226	SM	SM23	Alliance	Priest	0	4	8	518	10590	252	\N	1	heal	\N
3227	SM	SM23	Alliance	Mage	0	3	8	27174	6226	253	\N	1	dps	\N
3228	SM	SM23	Horde	Warlock	2	4	31	30722	14654	672	1	\N	dps	\N
3229	SM	SM23	Horde	Paladin	2	1	29	28115	14499	665	1	\N	dps	\N
3230	SM	SM23	Horde	Warlock	9	0	34	53066	27057	677	1	\N	dps	\N
3231	SM	SM23	Alliance	Paladin	7	2	12	56890	9513	271	\N	1	dps	\N
3232	SM	SM23	Horde	Death Knight	4	2	32	25546	8749	673	1	\N	dps	\N
3233	SM	SM23	Alliance	Demon Hunter	3	4	13	44863	15179	275	\N	1	dps	\N
3234	SM	SM23	Alliance	Warrior	1	5	11	32349	4873	266	\N	1	dps	\N
3235	SM	SM23	Horde	Shaman	0	0	35	8247	61285	680	1	\N	heal	\N
3236	SM	SM23	Horde	Mage	7	0	35	42833	2918	454	1	\N	dps	\N
3237	SM	SM23	Alliance	Druid	1	2	6	16486	2725	230	\N	1	dps	\N
3238	SM	SM23	Alliance	Hunter	0	6	9	15268	3613	254	\N	1	dps	\N
3239	SM	SM23	Alliance	Druid	0	6	10	13849	7618	262	\N	1	dps	\N
3240	SM	SM23	Horde	Paladin	6	2	36	47651	18027	459	1	\N	dps	\N
3241	SM	SM23	Horde	Rogue	4	1	34	28302	7932	677	1	\N	dps	\N
3242	SM	SM23	Horde	Druid	3	2	34	33443	7545	678	1	\N	dps	\N
3243	SM	SM23	Alliance	Paladin	2	5	8	32795	13170	251	\N	1	dps	\N
3244	SM	SM23	Horde	Paladin	2	3	27	7142	12549	666	1	\N	heal	\N
3245	SM	SM24	Alliance	Hunter	2	1	13	97002	11297	295	\N	1	dps	\N
3246	SM	SM24	Alliance	Shaman	1	6	7	30716	3985	277	\N	1	dps	\N
3247	SM	SM24	Alliance	Warrior	0	1	12	5268	0	293	\N	1	dps	\N
3248	SM	SM24	Horde	Warlock	2	1	34	54285	25145	520	1	\N	dps	\N
3249	SM	SM24	Horde	Shaman	1	1	35	10604	63349	522	1	\N	heal	\N
3250	SM	SM24	Horde	Priest	2	1	33	32042	55472	516	1	\N	heal	\N
3251	SM	SM24	Alliance	Shaman	0	8	7	11079	72041	277	\N	1	heal	\N
3252	SM	SM24	Alliance	Shaman	0	6	6	32107	12464	275	\N	1	dps	\N
3253	SM	SM24	Alliance	Rogue	1	4	12	30228	22017	293	\N	1	dps	\N
3254	SM	SM24	Horde	Priest	0	0	32	15845	56803	365	1	\N	heal	\N
3255	SM	SM24	Horde	Hunter	7	0	34	68432	288	518	1	\N	dps	\N
3256	SM	SM24	Horde	Druid	5	0	31	49055	6319	363	1	\N	dps	\N
3257	SM	SM24	Alliance	Death Knight	6	4	13	38865	17126	295	\N	1	dps	\N
3258	SM	SM24	Horde	Death Knight	7	3	31	53770	18224	511	1	\N	dps	\N
3259	SM	SM24	Horde	Shaman	5	2	32	41962	17458	514	1	\N	dps	\N
3260	SM	SM24	Alliance	Shaman	3	3	13	73537	15272	295	\N	1	dps	\N
3261	SM	SM24	Horde	Mage	3	2	31	53691	11602	365	1	\N	dps	\N
3262	SM	SM24	Horde	Demon Hunter	4	3	32	37494	6542	514	1	\N	dps	\N
3263	SM	SM24	Alliance	Druid	0	0	12	8353	3765	292	\N	1	dps	\N
3264	SM	SM24	Alliance	Priest	0	3	11	966	80305	288	\N	1	heal	\N
3265	SM	SM25	Horde	Paladin	2	4	12	40551	8052	341	1	\N	dps	\N
3266	SM	SM25	Alliance	Death Knight	4	1	25	62856	28173	262	\N	1	dps	\N
3267	SM	SM25	Alliance	Priest	0	1	25	8923	107000	262	\N	1	heal	\N
3268	SM	SM25	Horde	Warrior	4	3	14	41891	11939	495	1	\N	dps	\N
3269	SM	SM25	Alliance	Rogue	1	3	21	25928	4076	251	\N	1	dps	\N
3270	SM	SM25	Horde	Shaman	0	0	10	613	23247	336	1	\N	heal	\N
3271	SM	SM25	Horde	Shaman	0	3	11	9426	85812	488	1	\N	heal	\N
3272	SM	SM25	Horde	Hunter	2	2	11	31563	5674	339	1	\N	dps	\N
3273	SM	SM25	Alliance	Druid	1	1	15	31043	25938	236	\N	1	dps	\N
3274	SM	SM25	Horde	Hunter	1	1	11	25474	8929	489	1	\N	dps	\N
3275	SM	SM25	Alliance	Paladin	0	0	23	4557	50552	256	\N	1	heal	\N
3276	SM	SM25	Alliance	Demon Hunter	4	1	14	29853	5750	234	\N	1	dps	\N
3277	SM	SM25	Horde	Shaman	2	0	13	40232	5221	342	1	\N	dps	\N
3278	SM	SM25	Alliance	Warlock	1	0	4	274	0	165	\N	1	dps	\N
3279	SM	SM25	Horde	Warlock	0	0	11	23312	12891	338	1	\N	dps	\N
3280	SM	SM25	Alliance	Demon Hunter	1	0	9	15080	940	194	\N	1	dps	\N
3281	SM	SM25	Alliance	Hunter	2	3	21	40571	9671	248	\N	1	dps	\N
3282	SM	SM25	Horde	Death Knight	2	7	12	65907	25020	338	1	\N	dps	\N
3283	SM	SM25	Horde	Priest	1	5	13	49094	27286	344	1	\N	dps	\N
3284	SM	SM25	Alliance	Demon Hunter	5	0	25	54919	8624	262	\N	1	dps	\N
3285	TK	TK23	Alliance	Rogue	5	0	45	82690	4625	803	1	\N	dps	\N
3286	TK	TK23	Alliance	Mage	1	4	40	31427	10773	565	1	\N	dps	\N
3287	TK	TK23	Alliance	Paladin	9	1	43	102000	26818	798	1	\N	dps	\N
3288	TK	TK23	Horde	Hunter	0	1	7	11562	2654	109	\N	1	dps	\N
3289	TK	TK23	Horde	Warrior	1	4	16	46813	1761	143	\N	1	dps	\N
3290	TK	TK23	Horde	Shaman	0	5	12	4521	33390	135	\N	1	heal	\N
3291	TK	TK23	Alliance	Monk	0	0	45	1611	165000	803	1	\N	heal	\N
3292	TK	TK23	Horde	Priest	2	3	17	137000	43681	145	\N	1	dps	\N
3293	TK	TK23	Alliance	Warrior	4	4	37	43029	7148	554	1	\N	dps	\N
3294	TK	TK23	Alliance	Death Knight	9	0	45	74066	25583	803	1	\N	dps	\N
3295	TK	TK23	Horde	Mage	2	5	15	84153	18202	141	\N	1	dps	\N
3296	TK	TK23	Horde	Warrior	3	5	15	64704	7938	141	\N	1	dps	\N
3297	TK	TK23	Alliance	Rogue	1	2	40	18047	6606	567	1	\N	dps	\N
3298	TK	TK23	Horde	Warlock	3	5	15	74618	40979	141	\N	1	dps	\N
3299	TK	TK23	Alliance	Druid	0	2	45	1214	148000	578	1	\N	heal	\N
3300	TK	TK23	Alliance	Paladin	4	5	33	43566	16431	547	1	\N	dps	\N
3301	TK	TK23	Horde	Paladin	6	5	15	68941	13718	141	\N	1	dps	\N
3302	TK	TK23	Horde	Monk	0	5	15	7327	68031	141	\N	1	heal	\N
3303	TK	TK23	Horde	Priest	0	5	16	7564	56718	143	\N	1	heal	\N
3304	TK	TK23	Alliance	Paladin	10	0	45	112000	13972	578	1	\N	dps	\N
3305	TK	TK24	Alliance	Death Knight	4	4	7	38219	15746	165	\N	1	dps	\N
3306	TK	TK24	Horde	Druid	5	0	28	52396	2930	359	1	\N	dps	\N
3307	TK	TK24	Horde	Demon Hunter	3	1	27	46365	7870	357	1	\N	dps	\N
3308	TK	TK24	Alliance	Rogue	0	3	5	31066	1719	156	\N	1	dps	\N
3309	TK	TK24	Alliance	Druid	0	4	6	0	13970	161	\N	1	heal	\N
3310	TK	TK24	Horde	Shaman	5	1	25	42286	2712	503	1	\N	dps	\N
3311	TK	TK24	Alliance	Warlock	1	3	5	27484	13215	158	\N	1	dps	\N
3312	TK	TK24	Horde	Shaman	1	1	24	8657	52124	351	1	\N	heal	\N
3313	TK	TK24	Horde	Druid	2	3	25	33509	2088	353	1	\N	dps	\N
3314	TK	TK24	Horde	Druid	1	0	28	406	64450	509	1	\N	heal	\N
3315	TK	TK24	Horde	Shaman	1	1	27	9464	2763	507	1	\N	dps	\N
3316	TK	TK24	Alliance	Druid	0	2	7	2063	40445	166	\N	1	heal	\N
3317	TK	TK24	Horde	Warlock	5	0	28	10778	10348	509	1	\N	dps	\N
3318	TK	TK24	Alliance	Warlock	1	3	8	29871	8665	167	\N	1	dps	\N
3319	TK	TK24	Alliance	Hunter	0	4	9	31992	4936	172	\N	1	dps	\N
3320	TK	TK24	Alliance	Druid	3	3	7	37162	0	164	\N	1	dps	\N
3321	TK	TK24	Horde	Warlock	2	2	24	29638	11409	351	1	\N	dps	\N
3322	TK	TK24	Horde	Shaman	3	0	28	33035	2586	359	1	\N	dps	\N
3323	TP	TP17	Alliance	Mage	2	5	22	39242	9367	788	1	\N	dps	\N
3324	TP	TP17	Horde	Death Knight	2	3	26	51619	24932	160	\N	1	dps	\N
3325	TP	TP17	Alliance	Paladin	6	2	25	68171	34875	572	1	\N	dps	\N
3326	TP	TP17	Horde	Death Knight	3	5	24	53592	24945	158	\N	1	dps	\N
3327	TP	TP17	Horde	Demon Hunter	2	0	6	4097	364	94	\N	1	dps	\N
3328	TP	TP17	Horde	Rogue	3	2	20	25978	3870	148	\N	1	dps	\N
3329	TP	TP17	Horde	Warrior	9	3	26	73437	8642	167	\N	1	dps	\N
3330	TP	TP17	Alliance	Warrior	7	5	21	47444	16769	773	1	\N	dps	\N
3331	TP	TP17	Alliance	Priest	0	4	14	3526	67786	531	1	\N	heal	\N
3332	TP	TP17	Horde	Shaman	1	4	26	7115	36536	163	\N	1	heal	\N
3333	TP	TP17	Alliance	Demon Hunter	2	4	24	38766	8644	793	1	\N	dps	\N
3334	TP	TP17	Horde	Priest	1	3	27	12857	92572	165	\N	1	heal	\N
3335	TP	TP17	Horde	Shaman	2	0	26	58009	24073	164	\N	1	dps	\N
3336	TP	TP17	Alliance	Demon Hunter	1	0	15	13877	17792	534	1	\N	dps	\N
3337	TP	TP17	Alliance	Death Knight	5	5	25	74727	22558	799	1	\N	dps	\N
3338	TP	TP17	Horde	Mage	1	1	26	52507	5499	161	\N	1	dps	\N
3339	TP	TP17	Alliance	Priest	0	2	20	4501	31591	550	1	\N	heal	\N
3340	TP	TP17	Horde	Druid	1	2	12	26032	2450	117	\N	1	dps	\N
3341	TP	TP17	Alliance	Hunter	2	1	21	25127	5880	779	1	\N	dps	\N
3342	TP	TP17	Alliance	Druid	1	2	11	33671	2135	524	1	\N	dps	\N
3343	TP	TP18	Alliance	Warrior	6	5	12	47277	9852	262	\N	1	dps	\N
3344	TP	TP18	Alliance	Mage	1	7	7	50145	28130	242	\N	1	dps	\N
3345	TP	TP18	Horde	Rogue	4	3	52	90104	15484	455	1	\N	dps	\N
3346	TP	TP18	Horde	Priest	0	0	57	11663	138000	467	1	\N	heal	\N
3347	TP	TP18	Alliance	Druid	1	8	10	1578	147000	254	\N	1	heal	\N
3348	TP	TP18	Alliance	Warlock	1	8	9	75206	39038	250	\N	1	dps	\N
3349	TP	TP18	Alliance	Warlock	0	4	7	109000	49288	240	\N	1	dps	\N
3350	TP	TP18	Horde	Shaman	0	1	48	26491	118000	451	1	\N	heal	\N
3351	TP	TP18	Alliance	Monk	2	8	13	42610	36938	266	\N	1	dps	\N
3352	TP	TP18	Alliance	Priest	2	9	11	89016	33941	260	\N	1	dps	\N
3353	TP	TP18	Horde	Priest	10	4	47	157000	56731	453	1	\N	dps	\N
3354	TP	TP18	Alliance	Hunter	0	0	0	3941	178	142	\N	1	dps	\N
3355	TP	TP18	Horde	Death Knight	11	1	52	89787	16441	462	1	\N	dps	\N
3356	TP	TP18	Alliance	Demon Hunter	0	1	1	4913	1335	131	\N	1	dps	\N
3357	TP	TP18	Horde	Warlock	1	3	40	28747	23714	432	1	\N	dps	\N
3358	TP	TP18	Alliance	Hunter	0	5	3	20027	3637	197	\N	1	dps	\N
3359	TP	TP18	Horde	Druid	12	1	57	158000	13356	617	1	\N	dps	\N
3360	TP	TP18	Horde	Shaman	10	0	49	90549	7801	451	1	\N	dps	\N
3361	TP	TP18	Horde	Monk	9	1	40	69888	30905	443	1	\N	dps	\N
3362	TP	TP18	Horde	Death Knight	8	1	40	53469	19973	433	1	\N	dps	\N
3363	AB	AB11	Horde	Paladin	12	1	54	95674	23607	561	1	\N	dps	\N
3364	AB	AB11	Alliance	Mage	0	6	12	21334	4787	329	\N	1	dps	\N
3365	AB	AB11	Horde	Demon Hunter	2	1	41	16147	4255	391	1	\N	dps	\N
3366	AB	AB11	Alliance	Mage	4	6	16	61251	16135	340	\N	1	dps	\N
3367	AB	AB11	Alliance	Monk	0	2	4	3618	31740	316	\N	1	heal	\N
3368	AB	AB11	Horde	Shaman	3	4	41	26924	108000	543	1	\N	heal	\N
3369	AB	AB11	Horde	Death Knight	7	0	60	127000	20905	287	1	\N	dps	\N
3370	AB	AB11	Alliance	Rogue	3	3	14	38447	12744	335	\N	1	dps	\N
3371	AB	AB11	Horde	Mage	7	1	44	70343	4129	575	1	\N	dps	\N
3372	AB	AB11	Alliance	Death Knight	1	4	6	61488	61246	319	\N	1	dps	\N
3373	AB	AB11	Alliance	Hunter	0	6	12	120000	12055	327	\N	1	dps	\N
3374	AB	AB11	Horde	Druid	0	0	59	5372	79191	581	1	\N	heal	\N
3375	AB	AB11	Alliance	Paladin	0	8	11	101000	8543	324	\N	1	dps	\N
3376	AB	AB11	Alliance	Monk	3	5	11	73221	50848	345	\N	1	dps	\N
3377	AB	AB11	Alliance	Monk	0	4	5	19469	14544	237	\N	1	dps	\N
3378	AB	AB11	Horde	Rogue	3	4	17	70258	11655	524	1	\N	dps	\N
3379	AB	AB11	Alliance	Warlock	1	5	14	68921	10064	342	\N	1	dps	\N
3380	AB	AB11	Horde	Shaman	5	5	54	140000	13870	405	1	\N	dps	\N
3381	AB	AB11	Alliance	Warlock	6	6	13	115000	50470	335	\N	1	dps	\N
3382	AB	AB11	Horde	Death Knight	10	3	54	100000	26222	558	1	\N	dps	\N
3383	AB	AB11	Alliance	Rogue	3	2	8	33930	4180	328	\N	1	dps	\N
3384	AB	AB11	Horde	Priest	3	1	44	49191	160000	539	1	\N	heal	\N
3385	AB	AB11	Horde	Shaman	1	2	56	15056	183000	401	1	\N	heal	\N
3386	AB	AB11	Alliance	Warlock	5	4	21	177000	67751	360	\N	1	dps	\N
3387	AB	AB11	Horde	Paladin	7	1	56	50181	17131	436	1	\N	dps	\N
3388	AB	AB11	Horde	Paladin	2	1	50	29980	123000	582	1	\N	heal	\N
3389	AB	AB11	Horde	Warrior	7	4	36	101000	30597	375	1	\N	dps	\N
3390	AB	AB11	Alliance	Monk	0	6	12	8733	67223	339	\N	1	heal	\N
3391	AB	AB11	Horde	Druid	0	1	57	18015	12833	580	1	\N	dps	\N
3392	AB	AB11	Alliance	Priest	2	7	20	42191	130000	360	\N	1	heal	\N
3393	BG	BG33	Horde	Priest	0	1	23	16714	128000	370	1	\N	heal	\N
3394	BG	BG33	Alliance	Rogue	0	1	0	2277	1177	202	\N	1	dps	\N
3395	BG	BG33	Alliance	Hunter	0	4	1	22974	10899	205	\N	1	dps	\N
3396	BG	BG33	Alliance	Death Knight	4	5	7	83759	24597	240	\N	1	dps	\N
3397	BG	BG33	Alliance	Death Knight	0	3	4	45332	14176	233	\N	1	dps	\N
3398	BG	BG33	Alliance	Warlock	0	2	2	18863	15565	211	\N	1	dps	\N
3399	BG	BG33	Horde	Shaman	2	1	11	21257	7990	347	1	\N	dps	\N
3400	BG	BG33	Horde	Shaman	0	1	25	9666	56712	520	1	\N	heal	\N
3401	BG	BG33	Horde	Mage	11	0	30	72913	3994	386	1	\N	dps	\N
3402	BG	BG33	Horde	Rogue	3	2	19	48875	12151	509	1	\N	dps	\N
3403	BG	BG33	Horde	Warlock	4	0	20	33520	30546	360	1	\N	dps	\N
3404	BG	BG33	Alliance	Warrior	1	5	2	34095	11427	218	\N	1	dps	\N
3405	BG	BG33	Alliance	Warlock	1	3	2	70149	46467	210	\N	1	dps	\N
3406	BG	BG33	Horde	Druid	6	0	21	94964	10202	363	1	\N	dps	\N
3407	BG	BG33	Horde	Paladin	3	1	23	46384	15068	366	1	\N	dps	\N
3408	BG	BG33	Horde	Warlock	2	1	12	20383	7045	499	1	\N	dps	\N
3409	BG	BG33	Horde	Shaman	0	0	24	18160	61305	370	1	\N	heal	\N
3410	BG	BG33	Alliance	Paladin	1	1	2	21499	6360	168	\N	1	dps	\N
3411	BG	BG33	Alliance	Mage	0	0	0	17486	3513	202	\N	1	dps	\N
3412	BG	BG34	Alliance	Druid	5	0	30	71826	236	784	1	\N	dps	\N
3413	BG	BG34	Horde	Monk	1	1	4	52319	38382	143	\N	1	dps	\N
3414	BG	BG34	Alliance	Shaman	1	0	28	31529	85010	772	1	\N	heal	\N
3415	BG	BG34	Horde	Shaman	0	0	3	16606	3778	111	\N	1	dps	\N
3416	BG	BG34	Horde	Warlock	0	2	3	43464	12174	141	\N	1	dps	\N
3417	BG	BG34	Horde	Rogue	1	5	3	28011	1176	141	\N	1	dps	\N
3418	BG	BG34	Alliance	Paladin	1	0	26	3032	108000	764	1	\N	heal	\N
3419	BG	BG34	Alliance	Monk	6	1	27	67790	14150	769	1	\N	dps	\N
3420	BG	BG34	Horde	Shaman	0	7	3	16419	73464	141	\N	1	heal	\N
3421	BG	BG34	Alliance	Shaman	5	2	27	106000	16270	769	1	\N	dps	\N
3422	BG	BG34	Alliance	Druid	1	0	30	5011	119000	784	1	\N	heal	\N
3423	BG	BG34	Alliance	Druid	1	0	30	32690	3829	559	1	\N	dps	\N
3424	BG	BG34	Alliance	Hunter	8	0	30	89493	8159	559	1	\N	dps	\N
3425	BG	BG34	Horde	Rogue	0	4	2	12436	9776	139	\N	1	dps	\N
3426	BG	BG34	Alliance	Death Knight	2	0	4	19681	21962	496	1	\N	dps	\N
3427	BG	BG34	Horde	Death Knight	1	4	4	34195	4936	133	\N	1	dps	\N
3428	BG	BG34	Alliance	Hunter	4	1	28	53452	3194	773	1	\N	dps	\N
3429	BG	BG34	Horde	Druid	0	2	4	55026	29654	143	\N	1	dps	\N
3430	BG	BG34	Horde	Death Knight	0	5	3	71937	18736	141	\N	1	dps	\N
3431	BG	BG34	Horde	Druid	0	1	4	820	72920	143	\N	1	heal	\N
3432	DG	DG4	Alliance	Priest	7	2	43	146000	39844	585	1	\N	dps	\N
3433	DG	DG4	Alliance	Hunter	2	2	41	106000	11815	574	1	\N	dps	\N
3434	DG	DG4	Alliance	Druid	0	1	43	17395	192000	596	1	\N	heal	\N
3435	DG	DG4	Alliance	Druid	2	3	42	69850	18514	802	1	\N	dps	\N
3436	DG	DG4	Horde	Monk	0	0	5	195	42834	120	\N	1	heal	\N
3437	DG	DG4	Horde	Paladin	2	6	23	111000	27394	257	\N	1	dps	\N
3438	DG	DG4	Horde	Demon Hunter	1	5	21	20160	2291	263	\N	1	dps	\N
3439	DG	DG4	Alliance	Warlock	2	3	36	48667	20738	796	1	\N	dps	\N
3440	DG	DG4	Alliance	Shaman	14	4	43	131000	18197	807	1	\N	dps	\N
3441	DG	DG4	Alliance	Druid	3	2	34	82854	4483	574	1	\N	dps	\N
3442	DG	DG4	Horde	Hunter	9	3	27	60591	11823	286	\N	1	dps	\N
3443	DG	DG4	Alliance	Hunter	8	3	44	99671	13983	808	1	\N	dps	\N
3444	DG	DG4	Alliance	Priest	0	3	44	13343	108000	585	1	\N	heal	\N
3445	DG	DG4	Horde	Hunter	7	3	25	92607	13008	265	\N	1	dps	\N
3446	DG	DG4	Alliance	Warlock	3	2	41	85316	40312	802	1	\N	dps	\N
3447	DG	DG4	Horde	Druid	0	3	22	8215	153000	256	\N	1	heal	\N
3448	DG	DG4	Horde	Hunter	0	5	20	66474	12890	254	\N	1	dps	\N
3449	DG	DG4	Alliance	Demon Hunter	6	3	32	58659	20013	567	1	\N	dps	\N
3450	DG	DG4	Horde	Paladin	4	4	25	136000	20361	260	\N	1	dps	\N
3451	DG	DG4	Horde	Druid	3	1	22	35621	19489	276	\N	1	dps	\N
3452	DG	DG4	Horde	Monk	0	2	27	15464	241000	270	\N	1	heal	\N
3453	DG	DG4	Alliance	Druid	3	3	34	57157	8669	785	1	\N	dps	\N
3454	DG	DG4	Horde	Druid	2	3	19	41436	11302	191	\N	1	dps	\N
3455	DG	DG4	Horde	Shaman	2	5	24	129000	10799	258	\N	1	dps	\N
3456	DG	DG4	Horde	Shaman	0	2	24	573	106000	258	\N	1	heal	\N
3457	DG	DG4	Alliance	Druid	0	4	19	38108	8147	521	1	\N	dps	\N
3458	DG	DG4	Alliance	Shaman	0	1	44	11584	123000	820	1	\N	heal	\N
3459	DG	DG4	Horde	Priest	2	5	21	92870	24650	253	\N	1	dps	\N
3460	DG	DG4	Alliance	Mage	2	2	32	40445	26269	570	1	\N	dps	\N
3461	ES	ES14	Alliance	Mage	2	7	9	29411	10153	231	\N	1	dps	\N
3462	ES	ES14	Alliance	Warrior	1	3	10	22007	4419	235	\N	1	dps	\N
3463	ES	ES14	Alliance	Rogue	0	3	12	22881	5302	229	\N	1	dps	\N
3464	ES	ES14	Horde	Priest	0	0	57	7968	89357	508	1	\N	heal	\N
3465	ES	ES14	Horde	Demon Hunter	6	2	48	66780	20119	650	1	\N	dps	\N
3466	ES	ES14	Horde	Warrior	0	2	34	10040	4784	638	1	\N	dps	\N
3467	ES	ES14	Horde	Shaman	3	0	60	47251	15344	660	1	\N	dps	\N
3468	ES	ES14	Horde	Warrior	1	2	53	26009	27560	501	1	\N	dps	\N
3469	ES	ES14	Alliance	Rogue	1	8	11	51860	21382	229	\N	1	dps	\N
3470	ES	ES14	Horde	Paladin	3	2	38	30684	6895	491	1	\N	dps	\N
3471	ES	ES14	Horde	Priest	6	0	57	105000	29364	659	1	\N	dps	\N
3472	ES	ES14	Alliance	Priest	0	3	16	16943	183000	242	\N	1	heal	\N
3473	ES	ES14	Horde	Paladin	11	2	55	75810	21109	665	1	\N	dps	\N
3474	ES	ES14	Horde	Shaman	7	1	55	104000	12231	503	1	\N	dps	\N
3475	ES	ES14	Alliance	Priest	0	4	14	13302	92997	238	\N	1	heal	\N
3476	ES	ES14	Horde	Shaman	0	2	55	7738	97421	511	1	\N	heal	\N
3477	ES	ES14	Alliance	Demon Hunter	0	6	12	44667	7825	231	\N	1	dps	\N
3478	ES	ES14	Alliance	Warrior	1	5	14	53302	15319	240	\N	1	dps	\N
3479	ES	ES14	Alliance	Mage	2	3	11	19033	16219	235	\N	1	dps	\N
3480	ES	ES14	Horde	Death Knight	1	1	53	42748	20409	663	1	\N	dps	\N
3481	ES	ES14	Alliance	Priest	3	2	13	116000	30721	236	\N	1	dps	\N
3482	ES	ES14	Alliance	Hunter	1	7	11	51585	12988	231	\N	1	dps	\N
3483	ES	ES14	Horde	Druid	0	2	49	5128	52538	646	1	\N	heal	\N
3484	ES	ES14	Horde	Priest	6	1	41	37008	9052	644	1	\N	dps	\N
3485	ES	ES14	Alliance	Mage	0	1	0	6633	1210	142	\N	1	dps	\N
3486	ES	ES14	Horde	Shaman	8	1	56	80066	9329	653	1	\N	dps	\N
3487	ES	ES14	Horde	Mage	2	0	55	73018	3756	654	1	\N	dps	\N
3488	ES	ES14	Alliance	Warlock	5	6	10	83664	43580	230	\N	1	dps	\N
3489	ES	ES15	Horde	Priest	0	2	59	14530	193000	639	1	\N	heal	\N
3490	ES	ES15	Horde	Warrior	10	7	51	88288	10991	619	1	\N	dps	\N
3491	ES	ES15	Alliance	Warlock	7	3	47	104000	9936	418	\N	1	dps	\N
3492	ES	ES15	Alliance	Mage	1	5	41	41254	12295	420	\N	1	dps	\N
3493	ES	ES15	Alliance	Druid	1	2	46	7982	226000	446	\N	1	heal	\N
3494	ES	ES15	Alliance	Warlock	2	4	38	49345	15896	406	\N	1	dps	\N
3495	ES	ES15	Alliance	Mage	2	8	42	101000	27010	408	\N	1	dps	\N
3496	ES	ES15	Alliance	Hunter	2	5	42	56800	10894	415	\N	1	dps	\N
3497	ES	ES15	Horde	Shaman	0	4	46	16032	164000	622	1	\N	heal	\N
3498	ES	ES15	Horde	Demon Hunter	6	4	56	145000	21406	635	1	\N	dps	\N
3499	ES	ES15	Horde	Rogue	7	3	56	73026	18842	635	1	\N	dps	\N
3500	ES	ES15	Horde	Warlock	5	7	47	228000	92274	618	1	\N	dps	\N
3501	ES	ES15	Horde	Paladin	1	2	42	21206	93597	466	1	\N	heal	\N
3502	ES	ES15	Horde	Rogue	3	6	52	82185	18296	621	1	\N	dps	\N
3503	ES	ES15	Alliance	Warrior	6	4	36	40114	4843	372	\N	1	dps	\N
3504	ES	ES15	Alliance	Warlock	5	3	44	115000	44827	419	\N	1	dps	\N
3505	ES	ES15	Horde	Warlock	7	1	51	84700	36661	617	1	\N	dps	\N
3506	ES	ES15	Horde	Paladin	0	4	52	18200	17035	621	1	\N	dps	\N
3507	ES	ES15	Horde	Death Knight	9	6	54	141000	36031	637	1	\N	dps	\N
3508	ES	ES15	Alliance	Monk	0	3	44	5431	276000	415	\N	1	heal	\N
3509	ES	ES15	Alliance	Demon Hunter	2	7	42	55314	7761	417	\N	1	dps	\N
3510	ES	ES15	Alliance	Warrior	11	7	41	116000	29467	415	\N	1	dps	\N
3511	ES	ES15	Horde	Mage	8	3	56	82666	13862	640	1	\N	dps	\N
3512	ES	ES15	Alliance	Rogue	5	4	39	56148	20349	399	\N	1	dps	\N
3513	ES	ES15	Alliance	Hunter	8	3	49	200000	19679	430	\N	1	dps	\N
3514	ES	ES15	Horde	Mage	2	4	55	58913	7631	626	1	\N	dps	\N
3515	ES	ES15	Alliance	Rogue	1	5	35	31233	2387	401	\N	1	dps	\N
3516	ES	ES15	Horde	Demon Hunter	1	1	39	25934	3922	621	1	\N	dps	\N
3517	ES	ES15	Horde	Rogue	7	2	46	75119	7590	473	1	\N	dps	\N
3518	ES	ES15	Alliance	Druid	0	4	34	5623	70676	390	\N	1	heal	\N
3519	SA	SA4	Alliance	Death Knight	13	7	77	186000	48693	656	1	\N	dps	\N
3520	SA	SA4	Alliance	Hunter	2	6	70	58205	17034	623	1	\N	dps	\N
3521	SA	SA4	Horde	Mage	6	4	39	83233	9619	197	\N	1	dps	\N
3522	SA	SA4	Alliance	Paladin	9	3	69	154000	53475	614	1	\N	dps	\N
3523	SA	SA4	Alliance	Warrior	7	2	46	39960	7294	790	1	\N	dps	\N
3524	SA	SA4	Alliance	Hunter	9	1	68	145000	15349	624	1	\N	dps	\N
3525	SA	SA4	Horde	Mage	9	6	43	125000	24396	208	\N	1	dps	\N
3526	SA	SA4	Alliance	Warlock	8	1	71	136000	29221	687	1	\N	dps	\N
3527	SA	SA4	Horde	Hunter	2	5	41	145000	10732	186	\N	1	dps	\N
3528	SA	SA4	Alliance	Priest	2	3	66	54011	209000	614	1	\N	heal	\N
3529	SA	SA4	Alliance	Druid	5	6	64	193000	16712	614	1	\N	dps	\N
3530	SA	SA4	Horde	Hunter	4	4	33	79672	11254	171	\N	1	dps	\N
3531	SA	SA4	Horde	Priest	1	5	34	42579	133000	196	\N	1	heal	\N
3532	SA	SA4	Horde	Shaman	1	9	44	32136	151000	207	\N	1	heal	\N
3533	SA	SA4	Alliance	Druid	7	3	77	114000	16051	854	1	\N	dps	\N
3534	SA	SA4	Horde	Demon Hunter	9	3	39	125000	18580	200	\N	1	dps	\N
3535	SA	SA4	Horde	Death Knight	2	2	19	40455	13849	135	\N	1	dps	\N
3536	SA	SA4	Alliance	Mage	6	3	76	92956	13690	875	1	\N	dps	\N
3537	SA	SA4	Alliance	Death Knight	4	5	76	29652	24615	645	1	\N	dps	\N
3538	SA	SA4	Horde	Demon Hunter	7	8	40	157000	19520	201	\N	1	dps	\N
3539	SA	SA4	Horde	Mage	2	1	21	25913	579	140	\N	1	dps	\N
3540	SA	SA4	Alliance	Mage	5	4	67	111000	38155	847	1	\N	dps	\N
3541	SA	SA4	Horde	Warrior	0	1	15	33844	47712	130	\N	1	dps	\N
3542	SA	SA4	Horde	Druid	1	3	40	33489	189000	200	\N	1	heal	\N
3543	SA	SA4	Alliance	Druid	0	3	69	31732	150000	627	1	\N	heal	\N
3544	SA	SA4	Alliance	Mage	1	5	54	63203	8862	813	1	\N	dps	\N
3545	SA	SA4	Horde	Priest	0	6	39	1674	234000	201	\N	1	heal	\N
3546	SA	SA4	Horde	Warrior	0	4	23	62816	8733	145	\N	1	dps	\N
3547	SA	SA4	Alliance	Monk	0	3	78	726	223000	650	1	\N	heal	\N
3548	SA	SA4	Horde	Death Knight	3	7	34	128000	25293	196	\N	1	dps	\N
3549	SM	SM26	Alliance	Hunter	1	7	13	71930	15065	278	\N	1	dps	\N
3550	SM	SM26	Alliance	Hunter	6	2	17	55144	10838	298	\N	1	dps	\N
3551	SM	SM26	Horde	Shaman	4	0	35	78403	9634	528	1	\N	dps	\N
3552	SM	SM26	Alliance	Shaman	0	2	12	17566	12109	275	\N	1	dps	\N
3553	SM	SM26	Alliance	Druid	0	1	13	417	49145	285	\N	1	heal	\N
3554	SM	SM26	Horde	Shaman	0	2	36	9710	75893	529	1	\N	heal	\N
3555	SM	SM26	Alliance	Rogue	4	1	17	20908	17645	292	\N	1	dps	\N
3556	SM	SM26	Horde	Warrior	8	2	32	50497	15763	523	1	\N	dps	\N
3557	SM	SM26	Horde	Shaman	3	3	31	45926	10094	520	1	\N	dps	\N
3558	SM	SM26	Alliance	Paladin	3	5	14	74845	39561	283	\N	1	dps	\N
3559	SM	SM26	Horde	Shaman	5	0	36	62710	13916	379	1	\N	dps	\N
3560	SM	SM26	Horde	Priest	1	2	37	27624	71874	532	1	\N	heal	\N
3561	SM	SM26	Alliance	Warlock	3	6	15	45038	23964	284	\N	1	dps	\N
3562	SM	SM26	Horde	Shaman	7	1	36	76833	13540	529	1	\N	dps	\N
3563	SM	SM26	Horde	Rogue	1	3	29	23179	2964	518	1	\N	dps	\N
3564	SM	SM26	Alliance	Hunter	1	5	15	23957	9890	285	\N	1	dps	\N
3565	SM	SM26	Alliance	Paladin	0	5	13	2644	54330	280	\N	1	heal	\N
3566	SM	SM26	Horde	Hunter	3	4	27	39427	5921	514	1	\N	dps	\N
3567	SM	SM26	Horde	Warrior	7	1	36	36945	4682	190	1	\N	dps	\N
3568	SM	SM27	Alliance	Rogue	1	2	6	22267	10227	278	\N	1	dps	\N
3569	SM	SM27	Alliance	Rogue	0	5	5	27936	10576	274	\N	1	dps	\N
3570	SM	SM27	Horde	Paladin	0	0	26	16878	107000	360	1	\N	heal	\N
3571	SM	SM27	Alliance	Druid	0	4	8	4065	146000	287	\N	1	heal	\N
3572	SM	SM27	Horde	Priest	0	3	21	3335	76958	350	1	\N	heal	\N
3573	SM	SM27	Horde	Druid	4	0	26	121000	20637	360	1	\N	dps	\N
3574	SM	SM27	Alliance	Paladin	0	2	6	23786	13721	279	\N	1	dps	\N
3575	SM	SM27	Alliance	Shaman	3	1	5	33530	1877	250	\N	1	dps	\N
3576	SM	SM27	Horde	Warlock	5	1	21	33823	27244	508	1	\N	dps	\N
3577	SM	SM27	Alliance	Mage	0	3	7	55174	13978	283	\N	1	dps	\N
3578	SM	SM27	Horde	Shaman	0	1	25	23240	72178	508	1	\N	heal	\N
3579	SM	SM27	Horde	Druid	2	1	23	117000	4158	504	1	\N	dps	\N
3580	SM	SM27	Horde	Warrior	8	1	24	64703	9283	506	1	\N	dps	\N
3581	SM	SM27	Alliance	Druid	2	1	8	46208	6780	297	\N	1	dps	\N
3582	SM	SM27	Horde	Hunter	4	0	22	40768	7504	352	1	\N	dps	\N
3583	SM	SM27	Alliance	Mage	0	4	5	50969	8571	275	\N	1	dps	\N
3584	SM	SM27	Alliance	Priest	1	2	7	23690	99306	282	\N	1	heal	\N
3585	SM	SM27	Horde	Mage	1	2	20	31486	19549	348	1	\N	dps	\N
3586	SM	SM27	Alliance	Priest	0	1	0	10283	2194	157	\N	1	dps	\N
3587	SM	SM27	Horde	Warrior	1	1	24	21874	5318	356	1	\N	dps	\N
3588	TK	TK25	Horde	Mage	9	0	46	54490	6353	546	1	\N	dps	\N
3589	TK	TK25	Alliance	Warrior	1	7	23	25928	1338	337	\N	1	dps	\N
3590	TK	TK25	Alliance	Shaman	0	4	10	24232	7826	102	\N	1	dps	\N
3591	TK	TK25	Alliance	Paladin	5	5	25	75148	14018	345	\N	1	dps	\N
3592	TK	TK25	Horde	Shaman	1	4	43	21173	80209	540	1	\N	heal	\N
3593	TK	TK25	Horde	Demon Hunter	3	3	41	51042	8672	536	1	\N	dps	\N
3594	TK	TK25	Horde	Priest	1	0	46	3979	143000	546	1	\N	heal	\N
3595	TK	TK25	Alliance	Mage	1	1	30	86365	25244	361	\N	1	dps	\N
3596	TK	TK25	Horde	Demon Hunter	11	4	40	68546	9218	535	1	\N	dps	\N
3597	TK	TK25	Horde	Warlock	8	3	40	74327	27260	534	1	\N	dps	\N
3598	TK	TK25	Horde	Monk	0	2	42	923	98697	388	1	\N	heal	\N
3599	TK	TK25	Horde	Mage	6	6	36	71340	4595	526	1	\N	dps	\N
3600	TK	TK25	Horde	Priest	1	2	43	5642	67232	390	1	\N	heal	\N
3601	TK	TK25	Horde	Druid	4	5	36	55777	7151	526	1	\N	dps	\N
3602	TK	TK25	Alliance	Rogue	1	5	26	8452	10643	350	\N	1	dps	\N
3603	TK	TK25	Alliance	Hunter	8	5	25	129000	10846	346	\N	1	dps	\N
3604	TK	TK25	Alliance	Druid	1	6	20	37121	7107	330	\N	1	dps	\N
3605	TK	TK25	Alliance	Priest	2	3	29	41328	13275	358	\N	1	dps	\N
3606	TK	TK25	Alliance	Rogue	3	8	20	63752	10578	330	\N	1	dps	\N
3607	TK	TK25	Alliance	Monk	7	1	29	89522	39455	354	\N	1	dps	\N
3608	TP	TP19	Horde	Rogue	0	6	11	22883	3201	145	\N	1	dps	\N
3609	TP	TP19	Horde	Warlock	2	4	10	55228	17249	143	\N	1	dps	\N
3610	TP	TP19	Horde	Priest	2	4	10	81494	24812	143	\N	1	dps	\N
3611	TP	TP19	Horde	Warlock	1	2	11	60947	12899	145	\N	1	dps	\N
3612	TP	TP19	Horde	Hunter	1	3	6	23428	8274	135	\N	1	dps	\N
3613	TP	TP19	Horde	Shaman	0	5	9	17290	76949	141	\N	1	heal	\N
3614	TP	TP19	Alliance	Monk	0	2	25	263	121000	574	1	\N	heal	\N
3615	TP	TP19	Alliance	Hunter	8	0	26	82068	11183	575	1	\N	dps	\N
3616	TP	TP19	Horde	Paladin	0	1	11	17940	133000	145	\N	1	heal	\N
3617	TP	TP19	Alliance	Hunter	2	1	27	57626	13050	580	1	\N	dps	\N
3618	TP	TP19	Alliance	Monk	1	0	32	8438	104000	592	1	\N	heal	\N
3619	TP	TP19	Horde	Paladin	0	0	0	1166	12032	85	\N	1	heal	\N
3620	TP	TP19	Horde	Warrior	0	4	5	17016	4945	133	\N	1	dps	\N
3621	TP	TP19	Alliance	Rogue	4	2	25	49189	5540	571	1	\N	dps	\N
3622	TP	TP19	Horde	Shaman	5	3	10	72566	21299	143	\N	1	dps	\N
3623	TP	TP19	Alliance	Death Knight	3	1	31	94728	18165	813	1	\N	dps	\N
3624	TP	TP19	Alliance	Warrior	8	2	29	39532	6868	580	1	\N	dps	\N
3625	TP	TP19	Alliance	Paladin	0	2	28	17042	5988	583	1	\N	dps	\N
3626	TP	TP19	Alliance	Demon Hunter	4	0	32	81799	11032	817	1	\N	dps	\N
3627	TP	TP19	Alliance	Warlock	2	1	26	44164	24206	579	1	\N	dps	\N
3628	WG	WG31	Alliance	Warrior	0	3	6	9164	636	193	\N	1	dps	\N
3629	WG	WG31	Horde	Warrior	14	1	48	64805	9455	556	1	\N	dps	\N
3630	WG	WG31	Horde	Priest	10	0	52	56631	5181	415	1	\N	dps	\N
3631	WG	WG31	Horde	Druid	9	1	52	73112	5871	415	1	\N	dps	\N
3632	WG	WG31	Horde	Priest	1	2	48	9707	3672	557	1	\N	dps	\N
3633	WG	WG31	Alliance	Demon Hunter	0	3	2	11200	749	151	\N	1	dps	\N
3634	WG	WG31	Alliance	Rogue	0	1	2	10706	1982	151	\N	1	dps	\N
3635	WG	WG31	Alliance	Hunter	0	3	1	8391	4249	131	\N	1	dps	\N
3636	WG	WG31	Horde	Death Knight	2	1	31	11288	3322	527	1	\N	dps	\N
3637	WG	WG31	Alliance	Warrior	0	4	2	11740	327	136	\N	1	dps	\N
3638	WG	WG31	Horde	Shaman	0	0	28	9784	33472	521	1	\N	heal	\N
3639	WG	WG31	Alliance	Warlock	0	5	5	20616	11421	190	\N	1	dps	\N
3640	WG	WG31	Alliance	Druid	1	2	1	8825	679	131	\N	1	dps	\N
3641	WG	WG31	Horde	Monk	0	0	52	442	80825	415	1	\N	heal	\N
3642	WG	WG31	Horde	Priest	10	0	52	65660	4364	415	1	\N	dps	\N
3643	WG	WG31	Alliance	Hunter	3	4	6	27231	10866	193	\N	1	dps	\N
3644	WG	WG31	Horde	Mage	3	2	38	29408	7570	537	1	\N	dps	\N
3645	WG	WG31	Alliance	Hunter	0	4	1	12773	3513	147	\N	1	dps	\N
3646	WG	WG31	Horde	Death Knight	4	1	48	44704	12882	557	1	\N	dps	\N
3647	WG	WG32	Alliance	Warrior	2	2	45	62198	15738	835	1	\N	dps	\N
3648	WG	WG32	Alliance	Mage	6	2	41	53964	13859	826	1	\N	dps	\N
3649	WG	WG32	Alliance	Mage	2	2	41	89857	22827	821	1	\N	dps	\N
3650	WG	WG32	Alliance	Demon Hunter	6	0	39	65097	16181	823	1	\N	dps	\N
3651	WG	WG32	Alliance	Death Knight	2	0	45	76273	27807	834	1	\N	dps	\N
3652	WG	WG32	Alliance	Paladin	5	1	45	100000	22952	610	1	\N	dps	\N
3653	WG	WG32	Alliance	Warlock	7	1	43	131000	29459	601	1	\N	dps	\N
3654	WG	WG32	Horde	Druid	4	7	9	62211	1636	143	\N	1	dps	\N
3655	WG	WG32	Horde	Shaman	1	6	6	11856	47842	127	\N	1	heal	\N
3656	WG	WG32	Horde	Mage	0	4	6	3788	9701	137	\N	1	dps	\N
3657	WG	WG32	Horde	Paladin	0	5	4	28837	5500	123	\N	1	dps	\N
3658	WG	WG32	Horde	Death Knight	0	2	7	56258	17548	139	\N	1	dps	\N
3659	WG	WG32	Alliance	Monk	1	0	44	2996	64708	607	1	\N	heal	\N
3660	WG	WG32	Horde	Shaman	2	8	7	87382	7987	139	\N	1	dps	\N
3661	WG	WG32	Horde	Monk	0	2	9	579	206000	143	\N	1	heal	\N
3662	WG	WG32	Horde	Demon Hunter	0	2	7	12102	2249	139	\N	1	dps	\N
3663	WG	WG32	Horde	Priest	0	4	8	52123	8011	131	\N	1	dps	\N
3664	WG	WG32	Horde	Druid	0	3	9	3151	92324	143	\N	1	heal	\N
3665	WG	WG32	Alliance	Paladin	0	1	40	3060	47377	823	1	\N	heal	\N
3666	WG	WG32	Alliance	Warrior	12	0	46	54988	14657	838	1	\N	dps	\N
3667	WG	WG33	Horde	Monk	0	2	8	11602	141000	486	1	\N	heal	\N
3668	WG	WG33	Horde	Paladin	5	1	14	100000	23696	500	1	\N	dps	\N
3669	WG	WG33	Horde	Warlock	3	2	13	86509	56155	503	1	\N	dps	\N
3670	WG	WG33	Alliance	Death Knight	1	2	23	91941	21368	336	\N	1	dps	\N
3671	WG	WG33	Horde	Warlock	2	4	11	87058	65324	347	1	\N	dps	\N
3672	WG	WG33	Horde	Shaman	2	5	8	27921	128000	337	1	\N	heal	\N
3673	WG	WG33	Horde	Priest	1	0	16	18014	122000	506	1	\N	heal	\N
3674	WG	WG33	Alliance	Death Knight	2	1	23	120000	18291	335	\N	1	dps	\N
3675	WG	WG33	Horde	Demon Hunter	0	4	6	52462	56238	484	1	\N	dps	\N
3676	WG	WG33	Horde	Shaman	1	3	13	21779	86327	348	1	\N	heal	\N
3677	WG	WG33	Alliance	Paladin	1	0	13	6973	86937	191	\N	1	heal	\N
3678	WG	WG33	Horde	Monk	0	4	8	5871	124000	487	1	\N	heal	\N
3679	WG	WG33	Alliance	Warlock	1	3	22	137000	22392	332	\N	1	dps	\N
3680	WG	WG33	Alliance	Mage	8	0	24	124000	22516	339	\N	1	dps	\N
3681	WG	WG33	Alliance	Priest	2	0	23	41907	127000	337	\N	1	heal	\N
3682	WG	WG33	Alliance	Mage	2	4	14	80581	27412	319	\N	1	dps	\N
3683	WG	WG33	Alliance	Death Knight	3	2	22	96812	24983	336	\N	1	dps	\N
3684	WG	WG33	Alliance	Death Knight	1	4	23	69035	19307	336	\N	1	dps	\N
3685	WG	WG33	Horde	Hunter	5	4	11	82922	23097	346	1	\N	dps	\N
3686	WG	WG33	Alliance	Druid	6	3	23	185000	17901	337	\N	1	dps	\N
3687	WG	WG34	Horde	Demon Hunter	2	0	31	49668	22632	386	1	\N	dps	\N
3688	WG	WG34	Alliance	Priest	0	1	8	11519	6959	211	\N	1	dps	\N
3689	WG	WG34	Alliance	Paladin	0	2	8	23509	8402	214	\N	1	dps	\N
3690	WG	WG34	Horde	Mage	4	0	32	80971	4139	388	1	\N	dps	\N
3691	WG	WG34	Horde	Hunter	5	2	30	89563	10763	534	1	\N	dps	\N
3692	WG	WG34	Alliance	Mage	1	0	1	12533	0	114	\N	1	dps	\N
3693	WG	WG34	Alliance	Warlock	5	3	12	129000	25689	223	\N	1	dps	\N
3694	WG	WG34	Horde	Shaman	0	1	32	20106	112000	538	1	\N	heal	\N
3695	WG	WG34	Horde	Druid	1	1	32	2876	132000	388	1	\N	heal	\N
3696	WG	WG34	Alliance	Druid	0	0	1	0	0	114	\N	1	heal	\N
3697	WG	WG34	Horde	Paladin	1	0	32	86131	28150	388	1	\N	dps	\N
3698	WG	WG34	Alliance	Mage	4	5	11	98132	11725	221	\N	1	dps	\N
3699	WG	WG34	Alliance	Priest	0	6	11	19162	153000	219	\N	1	heal	\N
3700	WG	WG34	Alliance	Death Knight	1	3	11	42021	12106	223	\N	1	dps	\N
3701	WG	WG34	Alliance	Shaman	3	7	13	79618	9517	227	\N	1	dps	\N
3702	WG	WG34	Horde	Death Knight	0	2	23	16804	7303	520	1	\N	dps	\N
3703	WG	WG34	Horde	Paladin	8	1	32	90118	36964	538	1	\N	dps	\N
3704	WG	WG34	Horde	Warrior	7	4	31	55626	8886	536	1	\N	dps	\N
3705	WG	WG34	Alliance	Druid	0	2	13	401	180000	227	\N	1	heal	\N
3706	WG	WG34	Horde	Death Knight	4	3	26	103000	20866	526	1	\N	dps	\N
3707	WG	WG35	Horde	Shaman	3	5	38	63723	37338	213	\N	1	dps	\N
3708	WG	WG35	Alliance	Death Knight	6	5	44	83779	181000	827	1	\N	dps	\N
3709	WG	WG35	Alliance	Warlock	5	5	47	70581	32087	818	1	\N	dps	\N
3710	WG	WG35	Horde	Hunter	0	5	36	11866	4594	209	\N	1	dps	\N
3711	WG	WG35	Alliance	Demon Hunter	5	7	44	69758	18108	817	1	\N	dps	\N
3712	WG	WG35	Horde	Priest	7	3	39	182000	52486	215	\N	1	dps	\N
3713	WG	WG35	Alliance	Hunter	6	5	48	87598	15799	827	1	\N	dps	\N
3714	WG	WG35	Horde	Shaman	0	9	35	19566	124000	208	\N	1	heal	\N
3715	WG	WG35	Horde	Warlock	2	7	37	95218	41899	212	\N	1	dps	\N
3716	WG	WG35	Alliance	Paladin	5	6	45	79699	20202	594	1	\N	dps	\N
3717	WG	WG35	Horde	Paladin	9	4	41	92747	38256	220	\N	1	dps	\N
3718	WG	WG35	Horde	Warlock	5	3	43	156000	66252	224	\N	1	dps	\N
3719	WG	WG35	Horde	Warrior	7	5	39	108000	22040	216	\N	1	dps	\N
3720	WG	WG35	Horde	Death Knight	5	8	31	102000	18483	199	\N	1	dps	\N
3721	WG	WG35	Alliance	Priest	0	2	41	24324	169000	577	1	\N	heal	\N
3722	WG	WG35	Alliance	Druid	18	1	52	123000	49511	615	1	\N	dps	\N
3723	WG	WG35	Alliance	Monk	0	4	33	13087	9057	555	1	\N	dps	\N
3724	WG	WG35	Horde	Hunter	2	5	28	57365	6837	193	\N	1	dps	\N
3725	WG	WG35	Alliance	Death Knight	2	5	41	44997	40934	581	1	\N	dps	\N
3726	WG	WG35	Alliance	Monk	6	5	44	85443	106000	808	1	\N	heal	\N
3727	AB	AB12	Alliance	Shaman	5	4	69	68457	12137	487	\N	1	dps	\N
3728	AB	AB12	Horde	Druid	0	7	24	126000	46257	519	1	\N	dps	\N
3729	AB	AB12	Horde	Warrior	2	8	24	86035	8773	369	1	\N	dps	\N
3730	AB	AB12	Alliance	Warrior	4	4	71	73524	21605	494	\N	1	dps	\N
3731	AB	AB12	Alliance	Mage	5	4	75	65909	6205	488	\N	1	dps	\N
3732	AB	AB12	Alliance	Death Knight	10	4	76	204000	63072	488	\N	1	dps	\N
3733	AB	AB12	Alliance	Rogue	7	3	75	56338	13561	538	\N	1	dps	\N
3734	AB	AB12	Horde	Warlock	4	1	19	67137	22048	225	1	\N	dps	\N
3735	AB	AB12	Horde	Shaman	0	13	16	26754	154000	504	1	\N	heal	\N
3736	AB	AB12	Alliance	Hunter	2	5	63	41776	3982	477	\N	1	dps	\N
3737	AB	AB12	Horde	Warrior	12	7	29	66915	22825	390	1	\N	dps	\N
3738	AB	AB12	Alliance	Druid	0	2	81	9180	143000	509	\N	1	heal	\N
3739	AB	AB12	Horde	Mage	6	2	17	107000	7904	506	1	\N	dps	\N
3740	AB	AB12	Horde	Mage	2	4	27	142000	16004	529	1	\N	dps	\N
3741	AB	AB12	Alliance	Rogue	5	4	43	80324	15124	445	\N	1	dps	\N
3742	AB	AB12	Alliance	Hunter	3	2	77	61051	13175	489	\N	1	dps	\N
3743	AB	AB12	Horde	Priest	1	7	31	3147	181000	547	1	\N	heal	\N
3744	AB	AB12	Horde	Warrior	3	6	9	34269	4530	482	1	\N	dps	\N
3745	AB	AB12	Horde	Warrior	6	6	31	68005	14666	404	1	\N	dps	\N
3746	AB	AB12	Alliance	Warlock	10	1	90	183000	39760	522	\N	1	dps	\N
3747	AB	AB12	Alliance	Druid	0	2	82	6551	225000	500	\N	1	heal	\N
3748	AB	AB12	Alliance	Death Knight	20	2	86	185000	36006	503	\N	1	dps	\N
3749	AB	AB12	Alliance	Shaman	9	5	67	136000	14643	387	\N	1	dps	\N
3750	AB	AB12	Horde	Hunter	6	4	36	179000	14960	569	1	\N	dps	\N
3751	AB	AB12	Horde	Hunter	4	7	29	100000	20198	381	1	\N	dps	\N
3752	AB	AB12	Horde	Rogue	0	4	26	65115	22177	386	1	\N	dps	\N
3753	AB	AB12	Horde	Shaman	0	10	16	17355	183000	507	1	\N	heal	\N
3754	AB	AB12	Alliance	Druid	1	3	75	3997	237000	496	\N	1	heal	\N
3755	AB	AB12	Horde	Druid	2	7	22	63888	57394	521	1	\N	dps	\N
3756	AB	AB12	Alliance	Warlock	10	4	77	104000	35555	510	\N	1	dps	\N
3757	AB	AB13	Alliance	Death Knight	5	3	14	108000	27213	277	\N	1	dps	\N
3758	AB	AB13	Horde	Priest	3	1	14	25138	15790	508	1	\N	dps	\N
3759	AB	AB13	Alliance	Druid	2	3	8	39871	274	261	\N	1	dps	\N
3760	AB	AB13	Horde	Mage	5	0	37	72322	15157	538	1	\N	dps	\N
3761	AB	AB13	Alliance	Priest	1	4	10	31563	6453	261	\N	1	dps	\N
3762	AB	AB13	Horde	Druid	9	0	37	57843	18418	391	1	\N	dps	\N
3763	AB	AB13	Horde	Monk	3	0	29	27160	116000	371	1	\N	heal	\N
3764	AB	AB13	Alliance	Paladin	1	5	13	85252	18839	272	\N	1	dps	\N
3765	AB	AB13	Horde	Shaman	0	3	40	13100	116000	547	1	\N	heal	\N
3766	AB	AB13	Alliance	Priest	5	1	16	71937	13116	177	\N	1	dps	\N
3767	AB	AB13	Horde	Mage	1	1	27	76985	11331	221	1	\N	dps	\N
3768	AB	AB13	Horde	Paladin	2	0	37	43803	10841	251	1	\N	dps	\N
3769	AB	AB13	Horde	Hunter	11	0	43	67648	2013	398	1	\N	dps	\N
3770	AB	AB13	Alliance	Shaman	0	3	13	6572	97878	272	\N	1	heal	\N
3771	AB	AB13	Alliance	Hunter	0	0	3	6674	0	170	\N	1	dps	\N
3772	AB	AB13	Alliance	Hunter	1	4	8	29038	1815	254	\N	1	dps	\N
3773	AB	AB13	Horde	Death Knight	8	3	34	63508	28477	533	1	\N	dps	\N
3774	AB	AB13	Horde	Shaman	1	2	22	29623	7103	516	1	\N	dps	\N
3775	AB	AB13	Alliance	Paladin	0	4	6	49108	16589	258	\N	1	dps	\N
3776	AB	AB13	Alliance	Mage	2	3	8	26403	5306	269	\N	1	dps	\N
3777	AB	AB13	Horde	Druid	2	2	23	10334	63156	205	1	\N	heal	\N
3778	AB	AB13	Alliance	Shaman	2	8	10	31159	9296	257	\N	1	dps	\N
3779	AB	AB13	Horde	Hunter	2	2	40	47825	14203	246	1	\N	dps	\N
3780	AB	AB13	Horde	Rogue	4	3	34	39882	6073	526	1	\N	dps	\N
3781	AB	AB13	Horde	Monk	0	3	33	294	93664	382	1	\N	heal	\N
3782	AB	AB13	Alliance	Priest	0	0	3	62	18280	184	\N	1	heal	\N
3783	AB	AB13	Alliance	Paladin	0	1	11	4728	74235	264	\N	1	heal	\N
3784	AB	AB13	Alliance	Rogue	1	7	8	28463	5393	266	\N	1	dps	\N
3785	AB	AB13	Alliance	Shaman	0	4	12	82325	9831	267	\N	1	dps	\N
3786	BG	BG35	Horde	Warlock	6	2	35	104000	38472	231	\N	1	dps	\N
3787	BG	BG35	Horde	Paladin	0	3	32	2898	140000	222	\N	1	heal	\N
3788	BG	BG35	Alliance	Demon Hunter	6	4	24	101000	7023	788	1	\N	dps	\N
3789	BG	BG35	Horde	Paladin	9	3	35	100000	15098	225	\N	1	dps	\N
3790	BG	BG35	Horde	Warrior	8	7	28	90374	21495	212	\N	1	dps	\N
3791	BG	BG35	Alliance	Druid	0	6	25	703	131000	564	1	\N	heal	\N
3792	BG	BG35	Alliance	Demon Hunter	5	3	33	99647	15687	822	1	\N	dps	\N
3793	BG	BG35	Horde	Death Knight	0	3	34	79739	18414	226	\N	1	dps	\N
3794	BG	BG35	Horde	Hunter	5	2	26	62772	13831	219	\N	1	dps	\N
3795	BG	BG35	Horde	Shaman	1	8	18	13376	90168	192	\N	1	heal	\N
3796	BG	BG35	Horde	Paladin	5	4	38	54866	21101	236	\N	1	dps	\N
3797	BG	BG35	Alliance	Warrior	8	4	33	89737	17021	818	1	\N	dps	\N
3798	BG	BG35	Alliance	Paladin	2	6	26	24981	211000	794	1	\N	heal	\N
3799	BG	BG35	Alliance	Druid	3	5	30	86191	5533	586	1	\N	dps	\N
3800	BG	BG35	Alliance	Demon Hunter	7	4	23	90542	16036	562	1	\N	dps	\N
3801	BG	BG35	Horde	Death Knight	3	7	18	59398	21239	194	\N	1	dps	\N
3802	BG	BG35	Alliance	Druid	4	7	16	63332	14499	771	1	\N	dps	\N
3803	BG	BG35	Alliance	Rogue	7	4	36	65988	3077	835	1	\N	dps	\N
3804	BG	BG35	Alliance	Rogue	0	1	5	14672	2970	505	1	\N	dps	\N
3805	BG	BG35	Horde	Warrior	5	3	38	62559	15945	237	\N	1	dps	\N
3806	BG	BG36	Alliance	Death Knight	2	0	11	8599	5338	513	1	\N	dps	\N
3807	BG	BG36	Alliance	Paladin	0	1	32	3437	114000	785	1	\N	heal	\N
3808	BG	BG36	Alliance	Monk	3	3	28	38034	13064	774	1	\N	dps	\N
3809	BG	BG36	Horde	Rogue	0	2	8	30824	9434	149	\N	1	dps	\N
3810	BG	BG36	Alliance	Paladin	13	1	32	101000	11597	560	1	\N	dps	\N
3811	BG	BG36	Horde	Demon Hunter	2	5	9	76121	19734	151	\N	1	dps	\N
3812	BG	BG36	Horde	Shaman	0	6	8	15955	75526	149	\N	1	heal	\N
3813	BG	BG36	Alliance	Hunter	5	1	29	41045	3363	777	1	\N	dps	\N
3814	BG	BG36	Horde	Mage	0	1	2	23812	4653	109	\N	1	dps	\N
3815	BG	BG36	Horde	Rogue	1	3	9	31217	4630	151	\N	1	dps	\N
3816	BG	BG36	Alliance	Demon Hunter	5	2	23	58867	14987	754	1	\N	dps	\N
3817	BG	BG36	Horde	Warlock	1	2	8	37590	6848	149	\N	1	dps	\N
3818	BG	BG36	Alliance	Monk	4	0	30	50113	17307	555	1	\N	dps	\N
3819	BG	BG36	Horde	Druid	3	1	8	67881	4878	149	\N	1	dps	\N
3820	BG	BG36	Horde	Priest	3	3	10	35712	9473	153	\N	1	dps	\N
3821	BG	BG36	Alliance	Demon Hunter	1	0	31	89847	6296	781	1	\N	dps	\N
3822	BG	BG36	Horde	Rogue	0	3	10	26238	4443	153	\N	1	dps	\N
3823	BG	BG36	Horde	Priest	0	3	10	706	106000	153	\N	1	heal	\N
3824	BG	BG36	Alliance	Demon Hunter	1	2	28	35729	679	550	1	\N	dps	\N
3825	BG	BG36	Alliance	Druid	0	0	33	5359	125000	565	1	\N	heal	\N
3826	BG	BG37	Horde	Death Knight	14	1	49	73077	20667	414	1	\N	dps	\N
3827	BG	BG37	Alliance	Hunter	0	5	13	34044	4732	269	\N	1	dps	\N
3828	BG	BG37	Horde	Druid	3	0	45	35290	15545	408	1	\N	dps	\N
3829	BG	BG37	Horde	Warlock	2	3	30	27367	18778	520	1	\N	dps	\N
3830	BG	BG37	Alliance	Warrior	2	5	12	22015	6398	263	\N	1	dps	\N
3831	BG	BG37	Horde	Shaman	12	1	48	67376	21834	412	1	\N	dps	\N
3832	BG	BG37	Horde	Warlock	1	5	41	30101	12774	395	1	\N	dps	\N
3833	BG	BG37	Horde	Shaman	2	2	41	11573	10511	542	1	\N	dps	\N
3834	BG	BG37	Alliance	Demon Hunter	1	5	13	17471	8930	262	\N	1	dps	\N
3835	BG	BG37	Alliance	Warlock	1	5	13	14970	8744	257	\N	1	dps	\N
3836	BG	BG37	Horde	Shaman	5	3	34	52109	19058	528	1	\N	dps	\N
3837	BG	BG37	Horde	Rogue	5	6	37	35326	14365	387	1	\N	dps	\N
3838	BG	BG37	Alliance	Hunter	6	5	21	43163	3602	297	\N	1	dps	\N
3839	BG	BG37	Alliance	Warlock	3	7	16	34849	11671	275	\N	1	dps	\N
3840	BG	BG37	Alliance	Hunter	4	4	20	27281	3600	296	\N	1	dps	\N
3841	BG	BG37	Alliance	Death Knight	3	6	22	46429	25778	299	\N	1	dps	\N
3842	BG	BG37	Horde	Warlock	3	5	43	44939	17702	400	1	\N	dps	\N
3843	BG	BG37	Alliance	Hunter	3	5	16	39773	11244	275	\N	1	dps	\N
3844	BG	BG37	Horde	Demon Hunter	6	3	44	46668	28437	554	1	\N	dps	\N
3845	BG	BG37	Alliance	Death Knight	6	6	21	52881	46103	296	\N	1	dps	\N
3846	DG	DG5	Horde	Demon Hunter	1	4	22	31231	15319	504	1	\N	dps	\N
3847	DG	DG5	Horde	Mage	9	0	52	60210	23296	407	1	\N	dps	\N
3848	DG	DG5	Horde	Paladin	3	3	43	87841	30902	541	1	\N	dps	\N
3849	DG	DG5	Horde	Mage	5	0	50	44025	13600	407	1	\N	dps	\N
3850	DG	DG5	Alliance	Monk	0	4	16	22659	70728	235	\N	1	heal	\N
3851	DG	DG5	Horde	Death Knight	1	2	40	72378	19559	386	1	\N	dps	\N
3852	DG	DG5	Alliance	Shaman	0	4	14	695	69961	228	\N	1	heal	\N
3853	DG	DG5	Alliance	Paladin	1	5	22	102000	22898	251	\N	1	dps	\N
3854	DG	DG5	Alliance	Paladin	7	4	23	74917	34544	258	\N	1	dps	\N
3855	DG	DG5	Horde	Warrior	10	2	36	71376	19435	372	1	\N	dps	\N
3856	DG	DG5	Horde	Druid	4	4	35	26586	5576	368	1	\N	dps	\N
3857	DG	DG5	Alliance	Hunter	3	5	17	56566	7918	240	\N	1	dps	\N
3858	DG	DG5	Horde	Shaman	1	5	37	19317	84842	522	1	\N	heal	\N
3859	DG	DG5	Horde	Shaman	5	2	44	44642	10678	531	1	\N	dps	\N
3860	DG	DG5	Alliance	Demon Hunter	10	4	25	114000	9754	264	\N	1	dps	\N
3861	DG	DG5	Alliance	Druid	0	2	13	224	53571	224	\N	1	heal	\N
3862	DG	DG5	Horde	Warrior	1	1	16	9234	1925	313	1	\N	dps	\N
3863	DG	DG5	Horde	Mage	6	1	50	56240	8685	407	1	\N	dps	\N
3864	DG	DG5	Alliance	Death Knight	1	5	14	42348	19647	224	\N	1	dps	\N
3865	DG	DG5	Alliance	Paladin	2	5	16	34850	10297	251	\N	1	dps	\N
3866	DG	DG5	Alliance	Hunter	0	4	6	22436	5337	139	\N	1	dps	\N
3867	DG	DG5	Horde	Paladin	0	2	44	1248	125000	395	1	\N	heal	\N
3868	DG	DG5	Horde	Warlock	8	1	49	93554	17476	557	1	\N	dps	\N
3869	DG	DG5	Horde	Shaman	3	3	45	72514	8496	545	1	\N	dps	\N
3870	DG	DG5	Alliance	Priest	0	5	18	1507	148000	240	\N	1	heal	\N
3871	DG	DG5	Alliance	Warrior	3	4	18	25255	880	241	\N	1	dps	\N
3872	DG	DG5	Alliance	Warrior	2	4	10	34636	2657	214	\N	1	dps	\N
3873	DG	DG5	Horde	Demon Hunter	0	4	35	24861	534	517	1	\N	dps	\N
3874	DG	DG5	Alliance	Shaman	2	0	6	8984	0	206	\N	1	dps	\N
3875	DG	DG5	Alliance	Rogue	2	3	24	42569	7710	276	\N	1	dps	\N
3876	DG	DG6	Horde	Rogue	1	1	11	16197	2812	111	\N	1	dps	\N
3877	DG	DG6	Alliance	Priest	0	0	41	3044	48031	816	1	\N	heal	\N
3878	DG	DG6	Alliance	Priest	5	1	35	49109	7502	772	1	\N	dps	\N
3879	DG	DG6	Alliance	Priest	5	0	36	94247	23782	777	1	\N	dps	\N
3880	DG	DG6	Alliance	Hunter	0	1	40	23311	1972	813	1	\N	dps	\N
3881	DG	DG6	Horde	Death Knight	1	4	7	38369	16954	104	\N	1	dps	\N
3882	DG	DG6	Alliance	Priest	8	2	28	64955	18172	542	1	\N	dps	\N
3883	DG	DG6	Alliance	Monk	1	1	18	18122	1870	768	1	\N	dps	\N
3884	DG	DG6	Alliance	Hunter	2	0	42	26308	0	823	1	\N	dps	\N
3885	DG	DG6	Horde	Rogue	0	0	6	2411	23	102	\N	1	dps	\N
3886	DG	DG6	Horde	Shaman	3	3	10	53085	5966	108	\N	1	dps	\N
3887	DG	DG6	Alliance	Hunter	3	1	28	50876	1574	541	1	\N	dps	\N
3888	DG	DG6	Alliance	Demon Hunter	3	0	16	28335	6297	760	1	\N	dps	\N
3889	DG	DG6	Horde	Warlock	0	3	7	54579	32845	102	\N	1	dps	\N
3890	DG	DG6	Horde	Shaman	0	5	5	11543	81656	100	\N	1	heal	\N
3891	DG	DG6	Alliance	Druid	0	0	28	1449	125000	784	1	\N	heal	\N
3892	DG	DG6	Horde	Shaman	0	4	9	3077	105000	106	\N	1	heal	\N
3893	DG	DG6	Horde	Priest	0	1	4	5781	6073	91	\N	1	heal	\N
3894	DG	DG6	Horde	Shaman	1	1	10	39095	4742	108	\N	1	dps	\N
3895	DG	DG6	Horde	Mage	0	3	10	36084	9961	111	\N	1	dps	\N
3896	DG	DG6	Horde	Priest	1	3	8	56527	16457	106	\N	1	dps	\N
3897	DG	DG6	Alliance	Hunter	3	1	34	75285	5092	774	1	\N	dps	\N
3898	DG	DG6	Horde	Rogue	0	3	6	13948	10335	104	\N	1	dps	\N
3899	DG	DG6	Alliance	Rogue	5	0	35	45921	5037	772	1	\N	dps	\N
3900	DG	DG6	Alliance	Warrior	6	1	35	52889	10585	776	1	\N	dps	\N
3901	DG	DG6	Alliance	Warrior	4	2	33	37346	6317	548	1	\N	dps	\N
3902	DG	DG6	Horde	Rogue	0	5	6	23718	16165	103	\N	1	dps	\N
3903	DG	DG6	Horde	Warrior	1	2	5	11962	1951	90	\N	1	dps	\N
3904	DG	DG6	Horde	Shaman	0	3	6	47500	11327	103	\N	1	dps	\N
3905	DG	DG6	Alliance	Druid	1	0	34	655	114000	773	1	\N	heal	\N
3906	SM	SM28	Alliance	Death Knight	4	4	29	46414	19217	556	1	\N	dps	\N
3907	SM	SM28	Horde	Death Knight	7	2	28	104000	34998	219	\N	1	dps	\N
3908	SM	SM28	Alliance	Shaman	1	2	29	20372	79291	557	1	\N	heal	\N
3909	SM	SM28	Horde	Druid	0	1	29	48240	29674	222	\N	1	dps	\N
3910	SM	SM28	Horde	Rogue	1	4	27	59600	8308	215	\N	1	dps	\N
3911	SM	SM28	Horde	Shaman	4	4	23	37579	3402	209	\N	1	dps	\N
3912	SM	SM28	Alliance	Druid	3	3	29	55729	19326	780	1	\N	dps	\N
3913	SM	SM28	Alliance	Monk	3	0	34	54994	26212	577	1	\N	dps	\N
3914	SM	SM28	Horde	Shaman	0	5	23	12998	79425	208	\N	1	heal	\N
3915	SM	SM28	Horde	Warlock	3	4	27	55327	36991	217	\N	1	dps	\N
3916	SM	SM28	Alliance	Demon Hunter	12	2	33	81315	10460	800	1	\N	dps	\N
3917	SM	SM28	Horde	Warrior	1	6	21	16802	8411	203	\N	1	dps	\N
3918	SM	SM28	Alliance	Warlock	0	5	25	21782	23432	545	1	\N	dps	\N
3919	SM	SM28	Alliance	Paladin	4	3	28	75946	16173	784	1	\N	dps	\N
3920	SM	SM28	Horde	Shaman	10	3	28	74197	14695	219	\N	1	dps	\N
3921	SM	SM28	Alliance	Priest	1	2	32	32438	98708	796	1	\N	heal	\N
3922	SM	SM28	Alliance	Warrior	7	6	23	73409	30464	543	1	\N	dps	\N
3923	SM	SM28	Horde	Priest	2	4	26	59405	25173	214	\N	1	dps	\N
3924	SM	SM28	Horde	Priest	1	4	25	25761	58115	213	\N	1	heal	\N
3925	SM	SM28	Alliance	Hunter	2	3	27	41414	5750	776	1	\N	dps	\N
3926	SM	SM29	Alliance	Druid	0	3	15	0	66928	964	1	\N	heal	1
3927	SM	SM29	Alliance	Demon Hunter	7	0	28	73701	21588	663	1	\N	dps	1
3928	SM	SM29	Alliance	Priest	0	1	24	25296	111000	644	1	\N	heal	1
3929	SM	SM29	Alliance	Druid	3	1	25	40026	36321	655	1	\N	dps	1
3930	SM	SM29	Alliance	Death Knight	0	1	27	38078	58817	993	1	\N	dps	1
3931	SM	SM29	Alliance	Rogue	3	2	24	36389	5221	644	1	\N	dps	1
3932	SM	SM29	Horde	Warlock	1	0	11	96704	12228	233	\N	1	dps	1
3933	SM	SM29	Horde	Shaman	0	1	0	5697	18144	142	\N	1	heal	1
3934	SM	SM29	Horde	Death Knight	2	2	10	71775	18721	231	\N	1	dps	1
3935	SM	SM29	Horde	Death Knight	1	5	11	36228	13733	234	\N	1	dps	1
3936	SM	SM29	Alliance	Warrior	8	2	22	97126	21608	977	1	\N	dps	1
3937	SM	SM29	Alliance	Paladin	4	1	28	74696	39307	999	1	\N	dps	1
3938	SM	SM29	Horde	Paladin	1	0	12	22934	19422	236	\N	1	dps	1
3939	SM	SM29	Alliance	Druid	1	1	22	27006	6452	639	1	\N	dps	1
3940	SM	SM29	Alliance	Rogue	3	0	29	67338	8606	1003	1	\N	dps	1
3941	SM	SM29	Horde	Death Knight	1	4	11	75329	18949	234	\N	1	dps	1
3942	SM	SM29	Horde	Monk	1	3	11	8766	100000	233	\N	1	heal	1
3943	SM	SM29	Horde	Hunter	2	4	11	29406	8871	234	\N	1	dps	1
3944	SM	SM29	Horde	Demon Hunter	2	5	9	67620	24024	229	\N	1	dps	1
3945	SM	SM29	Horde	Priest	0	2	10	11692	88207	232	\N	1	heal	1
3946	SM	SM30	Alliance	Warlock	2	3	18	52491	21481	357	\N	1	dps	1
3947	SM	SM30	Alliance	Shaman	1	2	18	33614	14559	351	\N	1	dps	1
3948	SM	SM30	Horde	Paladin	0	3	12	20547	13390	415	1	\N	dps	1
3949	SM	SM30	Horde	Paladin	1	5	17	88020	24392	425	1	\N	dps	1
3950	SM	SM30	Alliance	Warlock	10	3	20	92733	60297	359	\N	1	dps	1
3951	SM	SM30	Horde	Shaman	0	2	20	13276	86882	656	1	\N	heal	1
3952	SM	SM30	Alliance	Rogue	1	1	13	38643	2700	336	\N	1	dps	1
3953	SM	SM30	Alliance	Monk	1	0	20	2159	118000	358	\N	1	heal	1
3954	SM	SM30	Alliance	Rogue	1	3	15	41647	1042	342	\N	1	dps	1
3955	SM	SM30	Horde	Paladin	1	0	17	35969	28741	650	1	\N	dps	1
3956	SM	SM30	Horde	Druid	5	2	20	67594	621	431	1	\N	dps	1
3957	SM	SM30	Horde	Hunter	6	2	20	89300	3024	431	1	\N	dps	1
3958	SM	SM30	Alliance	Druid	1	2	14	1789	105000	340	\N	1	heal	1
3959	SM	SM30	Horde	Rogue	2	0	19	37654	8491	429	1	\N	dps	1
3960	SM	SM30	Horde	Demon Hunter	2	1	19	56612	7518	429	1	\N	dps	1
3961	SM	SM30	Alliance	Demon Hunter	2	3	19	52988	8123	354	\N	1	dps	1
3962	SM	SM30	Horde	Warrior	1	6	13	29730	3175	642	1	\N	dps	1
3963	SM	SM30	Horde	Monk	0	2	19	577	127000	429	1	\N	heal	1
3964	SM	SM30	Alliance	Hunter	2	0	21	42614	4805	364	\N	1	dps	1
3965	SM	SM30	Alliance	Mage	2	3	18	72038	10765	352	\N	1	dps	1
3966	SM	SM31	Horde	Paladin	3	5	34	79387	23139	247	\N	1	dps	\N
3967	SM	SM31	Alliance	Shaman	2	5	24	18538	10957	759	1	\N	dps	\N
3968	SM	SM31	Alliance	Demon Hunter	6	3	26	54396	18409	541	1	\N	dps	\N
3969	SM	SM31	Horde	Warlock	4	2	36	63824	20400	252	\N	1	dps	\N
3970	SM	SM31	Horde	Rogue	1	2	41	43946	12397	264	\N	1	dps	\N
3971	SM	SM31	Alliance	Death Knight	2	6	24	42361	5312	537	1	\N	dps	\N
3972	SM	SM31	Horde	Shaman	0	6	27	17410	76331	234	\N	1	heal	\N
3973	SM	SM31	Horde	Druid	2	2	36	50098	11640	253	\N	1	dps	\N
3974	SM	SM31	Alliance	Druid	4	3	28	23223	16265	773	1	\N	dps	\N
3975	SM	SM31	Alliance	Shaman	4	5	30	49644	8022	777	1	\N	dps	\N
3976	SM	SM31	Alliance	Death Knight	4	3	29	41129	6890	782	1	\N	dps	\N
3977	SM	SM31	Alliance	Paladin	2	4	24	44383	18195	758	1	\N	dps	\N
3978	SM	SM31	Horde	Warrior	11	2	37	55388	18266	250	\N	1	dps	\N
3979	SM	SM31	Alliance	Priest	0	3	28	20104	187000	774	1	\N	heal	\N
3980	SM	SM31	Horde	Hunter	5	4	37	79233	13187	251	\N	1	dps	\N
3981	SM	SM31	Alliance	Rogue	3	1	31	39909	10916	558	1	\N	dps	\N
3982	SM	SM31	Alliance	Warrior	5	7	24	69268	11930	760	1	\N	dps	\N
3983	SM	SM31	Horde	Mage	2	2	37	43340	9089	253	\N	1	dps	\N
3984	SM	SM31	Horde	Warrior	8	5	37	64925	14865	255	\N	1	dps	\N
3985	SM	SM31	Horde	Warrior	3	2	27	21125	2785	233	\N	1	dps	\N
3986	SS	SS1	Alliance	Mage	3	1	25	18983	2559	988	1	\N	dps	\N
3987	SS	SS1	Horde	Death Knight	0	3	2	18961	12088	129	\N	1	dps	\N
3988	SS	SS1	Alliance	Druid	0	0	22	11712	20971	978	1	\N	heal	\N
3989	SS	SS1	Horde	Shaman	0	4	2	9570	50375	129	\N	1	heal	\N
3990	SS	SS1	Alliance	Death Knight	1	0	21	18582	16514	985	1	\N	dps	\N
3991	SS	SS1	Alliance	Paladin	6	0	27	47882	10956	997	1	\N	dps	\N
3992	SS	SS1	Alliance	Druid	0	1	21	2546	38005	973	1	\N	heal	\N
3993	SS	SS1	Alliance	Rogue	4	0	21	23460	2043	977	1	\N	dps	\N
3994	SS	SS1	Horde	Paladin	1	2	3	10657	13873	131	\N	1	heal	\N
3995	SS	SS1	Alliance	Hunter	0	0	22	6499	2361	978	1	\N	dps	\N
3996	SS	SS1	Horde	Druid	0	4	2	24926	6241	129	\N	1	dps	\N
3997	SS	SS1	Horde	Warrior	0	2	1	5600	471	127	\N	1	dps	\N
3998	SS	SS1	Horde	Hunter	0	2	0	6817	1893	125	\N	1	dps	\N
3999	SS	SS1	Alliance	Priest	7	0	27	73614	21771	772	1	\N	dps	\N
4000	SS	SS1	Horde	Hunter	1	3	3	14415	4740	131	\N	1	dps	\N
4001	SS	SS1	Alliance	Death Knight	5	0	26	26243	6251	993	1	\N	dps	\N
4002	SS	SS1	Horde	Paladin	0	3	3	27011	7361	131	\N	1	dps	\N
4003	SS	SS1	Horde	Hunter	1	3	3	21616	5783	131	\N	1	dps	\N
4004	SS	SS1	Horde	Warrior	0	1	0	7214	566	85	\N	1	dps	\N
4005	SS	SS1	Alliance	Hunter	0	1	23	25346	3250	762	1	\N	dps	\N
4006	SS	SS2	Horde	Rogue	5	0	31	24019	3994	673	1	\N	dps	\N
4007	SS	SS2	Alliance	Druid	0	6	2	9812	5017	313	\N	1	dps	\N
4008	SS	SS2	Horde	Warlock	0	3	22	23058	5782	655	1	\N	dps	\N
4009	SS	SS2	Alliance	Druid	0	3	12	5460	68361	341	\N	1	heal	\N
4010	SS	SS2	Alliance	Hunter	0	2	12	25462	1785	341	\N	1	dps	\N
4011	SS	SS2	Horde	Shaman	0	1	28	8938	39199	666	1	\N	heal	\N
4012	SS	SS2	Horde	Paladin	7	0	29	74219	12448	668	1	\N	dps	\N
4013	SS	SS2	Alliance	Demon Hunter	0	3	8	12855	0	328	\N	1	dps	\N
4014	SS	SS2	Alliance	Death Knight	1	2	8	36184	36415	327	\N	1	dps	\N
4015	SS	SS2	Alliance	Shaman	1	4	12	43677	12207	341	\N	1	dps	\N
4016	SS	SS2	Horde	Shaman	6	1	30	49165	12107	521	1	\N	dps	\N
4017	SS	SS2	Horde	Shaman	0	0	29	5654	75041	518	1	\N	heal	\N
4018	SS	SS2	Alliance	Druid	3	4	8	45164	4224	328	\N	1	dps	\N
4019	SS	SS2	Horde	Warlock	8	0	29	72909	33505	518	1	\N	dps	\N
4020	SS	SS2	Alliance	Shaman	1	2	10	40400	11782	333	\N	1	dps	\N
4021	SS	SS2	Horde	Paladin	4	2	26	38886	8513	513	1	\N	dps	\N
4022	SS	SS2	Horde	Paladin	2	1	28	40049	17434	517	1	\N	dps	\N
4023	SS	SS2	Alliance	Warlock	3	3	11	43222	23118	338	\N	1	dps	\N
4024	SS	SS2	Horde	Warlock	0	4	18	17197	14095	646	1	\N	dps	\N
4025	SS	SS2	Alliance	Hunter	3	4	11	25390	4349	338	\N	1	dps	\N
4026	SS	SS3	Horde	Druid	8	2	48	72172	14621	380	\N	1	dps	1
4027	SS	SS3	Alliance	Warlock	3	6	24	60541	35613	1229	1	\N	dps	1
4028	SS	SS3	Alliance	Warrior	8	4	27	80872	26021	895	1	\N	dps	1
4029	SS	SS3	Horde	Rogue	6	6	35	34069	5596	466	\N	1	dps	1
4030	SS	SS3	Alliance	Priest	0	4	24	24079	66847	884	1	\N	heal	1
4031	SS	SS3	Horde	Shaman	1	4	35	13709	90141	446	\N	1	heal	1
4032	SS	SS3	Alliance	Mage	1	4	23	41015	8243	886	1	\N	dps	1
4033	SS	SS3	Alliance	Priest	0	4	30	6142	124000	910	1	\N	heal	1
4034	SS	SS3	Horde	Rogue	3	6	29	55131	13724	454	\N	1	dps	1
4035	SS	SS3	Horde	Druid	0	2	46	646	128000	376	\N	1	heal	1
4036	SS	SS3	Horde	Death Knight	5	4	45	107000	38420	486	\N	1	dps	1
4037	SS	SS3	Alliance	Demon Hunter	5	7	16	52927	7381	856	1	\N	dps	1
4038	SS	SS3	Alliance	Hunter	2	5	24	79689	13014	892	1	\N	dps	1
4039	SS	SS3	Alliance	Death Knight	7	6	28	130000	29799	900	1	\N	dps	1
4040	SS	SS3	Alliance	Warlock	2	5	24	33359	23717	890	1	\N	dps	1
4041	SS	SS3	Horde	Warlock	5	3	45	70824	38067	486	\N	1	dps	1
4042	SS	SS3	Horde	Paladin	7	1	50	71404	26570	496	\N	1	dps	1
4043	SS	SS3	Alliance	Druid	4	6	22	78396	10610	1220	1	\N	dps	1
4044	SS	SS3	Horde	Priest	8	4	44	139000	39809	484	\N	1	dps	1
4045	SS	SS4	Horde	Priest	3	2	20	52218	23854	507	1	\N	dps	\N
4046	SS	SS4	Alliance	Hunter	1	2	10	26369	4916	333	\N	1	dps	\N
4047	SS	SS4	Alliance	Paladin	0	5	6	3522	31583	319	\N	1	heal	\N
4048	SS	SS4	Alliance	Warrior	0	5	9	35443	13197	328	\N	1	dps	\N
4049	SS	SS4	Horde	Demon Hunter	3	1	26	44303	4843	519	1	\N	dps	\N
4050	SS	SS4	Horde	Shaman	0	3	25	11846	52272	667	1	\N	heal	\N
4051	SS	SS4	Alliance	Mage	2	3	10	20394	4053	333	\N	1	dps	\N
4052	SS	SS4	Horde	Priest	5	1	25	44062	11772	517	1	\N	dps	\N
4053	SS	SS4	Alliance	Shaman	3	2	13	32348	5501	343	\N	1	dps	\N
4054	SS	SS4	Alliance	Druid	2	3	13	42602	8904	343	\N	1	dps	\N
4055	SS	SS4	Alliance	Paladin	1	2	10	22461	12895	333	\N	1	dps	\N
4056	SS	SS4	Horde	Priest	0	1	21	4937	41267	510	1	\N	heal	\N
4057	SS	SS4	Horde	Mage	2	1	26	17937	159	671	1	\N	dps	\N
4058	SS	SS4	Alliance	Paladin	2	2	12	15625	17184	338	\N	1	heal	\N
4059	SS	SS4	Horde	Warlock	2	1	19	27293	19674	656	1	\N	dps	\N
4060	SS	SS4	Alliance	Mage	2	2	12	33835	6429	338	\N	1	dps	\N
4061	SS	SS4	Alliance	Shaman	0	4	9	2574	23333	332	\N	1	heal	\N
4062	SS	SS4	Horde	Warlock	5	2	25	35404	19317	668	1	\N	dps	\N
4063	SS	SS4	Horde	Hunter	6	1	21	34791	1575	660	1	\N	dps	\N
4064	SS	SS4	Horde	Mage	4	0	21	30937	4151	509	1	\N	dps	\N
4065	SS	SS5	Alliance	Hunter	4	4	16	54755	10169	354	\N	1	dps	\N
4066	SS	SS5	Horde	Shaman	0	4	27	12895	80812	522	1	\N	heal	\N
4067	SS	SS5	Alliance	Monk	1	3	14	24057	14599	464	\N	1	dps	\N
4068	SS	SS5	Alliance	Warrior	6	5	20	36555	9149	482	\N	1	dps	\N
4069	SS	SS5	Alliance	Paladin	3	3	20	61445	16100	480	\N	1	dps	\N
4070	SS	SS5	Alliance	Monk	0	3	17	2799	137000	467	\N	1	heal	\N
4071	SS	SS5	Alliance	Mage	3	2	19	48704	10466	474	\N	1	dps	\N
4072	SS	SS5	Horde	Priest	2	5	18	35854	14808	503	1	\N	dps	\N
4073	SS	SS5	Horde	Hunter	0	2	27	48998	8304	671	1	\N	dps	\N
4074	SS	SS5	Alliance	Paladin	3	3	20	48489	15025	478	\N	1	dps	\N
4075	SS	SS5	Horde	Priest	3	2	25	65066	11760	670	1	\N	dps	\N
4076	SS	SS5	Alliance	Mage	1	6	13	20937	9241	459	\N	1	dps	\N
4077	SS	SS5	Horde	Mage	1	2	23	15155	1606	666	1	\N	dps	\N
4078	SS	SS5	Horde	Warrior	7	1	27	45936	11892	673	1	\N	dps	\N
4079	SS	SS5	Alliance	Rogue	1	2	19	46488	7809	474	\N	1	dps	\N
4080	SS	SS5	Horde	Mage	5	2	29	45191	16677	528	1	\N	dps	\N
4081	SS	SS5	Horde	Druid	1	2	26	7754	84643	669	1	\N	heal	\N
4082	SS	SS5	Horde	Death Knight	5	1	28	69220	32772	673	1	\N	dps	\N
4083	SS	SS5	Horde	Priest	9	1	29	60272	17169	678	1	\N	dps	\N
4084	SS	SS5	Alliance	Hunter	0	2	19	22689	4764	474	\N	1	dps	\N
4085	TK	TK26	Alliance	Hunter	1	7	23	28927	7751	355	\N	1	dps	\N
4086	TK	TK26	Horde	Shaman	3	2	52	59230	4158	549	1	\N	dps	\N
4087	TK	TK26	Horde	Priest	1	3	30	6129	57789	322	1	\N	heal	\N
4088	TK	TK26	Alliance	Hunter	7	3	27	143000	15719	371	\N	1	dps	\N
4089	TK	TK26	Alliance	Warrior	7	7	22	147000	17062	345	\N	1	dps	\N
4090	TK	TK26	Alliance	Paladin	6	6	26	142000	34656	368	\N	1	dps	\N
4091	TK	TK26	Horde	Demon Hunter	14	2	56	105000	43579	557	1	\N	dps	\N
4092	TK	TK26	Alliance	Warlock	1	6	22	59717	30726	352	\N	1	dps	\N
4093	TK	TK26	Alliance	Priest	0	7	24	14383	99617	361	\N	1	heal	\N
4094	TK	TK26	Horde	Shaman	2	3	54	31536	139000	553	1	\N	heal	\N
4095	TK	TK26	Horde	Monk	0	2	58	2789	166000	411	1	\N	heal	\N
4096	TK	TK26	Horde	Paladin	5	3	51	39532	20908	547	1	\N	dps	\N
4097	TK	TK26	Horde	Death Knight	9	4	51	74487	48343	547	1	\N	dps	\N
4098	TK	TK26	Alliance	Priest	0	5	22	6832	49159	352	\N	1	heal	\N
4099	TK	TK26	Horde	Shaman	8	4	54	93782	9998	403	1	\N	dps	\N
4100	TK	TK26	Alliance	Warrior	2	8	16	30408	3860	339	\N	1	dps	\N
4101	TK	TK26	Alliance	Hunter	1	6	22	60308	12021	353	\N	1	dps	\N
4102	TK	TK26	Horde	Warlock	9	2	56	87233	30989	557	1	\N	dps	\N
4103	TK	TK26	Horde	Warlock	5	0	59	94488	34179	413	1	\N	dps	\N
4104	TK	TK27	Alliance	Warrior	7	0	26	43550	5535	532	1	\N	dps	\N
4105	TK	TK27	Horde	Warrior	1	4	2	47519	1381	99	\N	1	dps	\N
4106	TK	TK27	Alliance	Warrior	1	1	26	15274	1913	757	1	\N	dps	\N
4107	TK	TK27	Horde	Druid	0	3	1	22694	5585	97	\N	1	dps	\N
4108	TK	TK27	Alliance	Mage	5	0	26	31397	7194	757	1	\N	dps	\N
4109	TK	TK27	Alliance	Demon Hunter	2	1	23	25568	9383	526	1	\N	dps	\N
4110	TK	TK27	Horde	Druid	0	2	1	19993	1455	97	\N	1	dps	\N
4111	TK	TK27	Horde	Shaman	0	3	2	10236	40293	99	\N	1	heal	\N
4112	TK	TK27	Horde	Paladin	0	3	2	0	31435	99	\N	1	heal	\N
4113	TK	TK27	Horde	Druid	0	1	1	44736	3767	97	\N	1	dps	\N
4114	TK	TK27	Alliance	Monk	0	0	26	3	122000	532	1	\N	heal	\N
4115	TK	TK27	Horde	Mage	0	2	1	13191	1934	12	\N	1	dps	\N
4116	TK	TK27	Alliance	Mage	3	0	26	44885	2218	757	1	\N	dps	\N
4117	TK	TK27	Alliance	Priest	0	0	26	2611	42207	757	1	\N	heal	\N
4118	TK	TK27	Horde	Priest	1	2	2	47462	14893	99	\N	1	dps	\N
4119	TK	TK27	Alliance	Druid	2	0	26	43001	9542	532	1	\N	dps	\N
4120	TK	TK27	Alliance	Death Knight	4	0	26	31327	6114	532	1	\N	dps	\N
4121	TK	TK27	Alliance	Rogue	2	0	26	15547	6132	757	1	\N	dps	\N
4122	TK	TK27	Horde	Demon Hunter	0	3	1	8578	3794	97	\N	1	dps	\N
4123	TK	TK28	Horde	Demon Hunter	8	0	36	62110	4091	528	1	\N	dps	\N
4124	TK	TK28	Alliance	Monk	0	2	4	0	19846	153	\N	1	heal	\N
4125	TK	TK28	Horde	Druid	1	0	26	13733	0	508	1	\N	dps	\N
4126	TK	TK28	Horde	Hunter	6	0	36	28133	361	378	1	\N	dps	\N
4127	TK	TK28	Horde	Warrior	3	1	33	23615	756	366	1	\N	dps	\N
4128	TK	TK28	Alliance	Priest	0	3	0	0	35978	142	\N	1	heal	\N
4129	TK	TK28	Alliance	Warrior	3	3	3	35264	7593	149	\N	1	dps	\N
4130	TK	TK28	Horde	Druid	3	1	32	14690	8530	520	1	\N	dps	\N
4131	TK	TK28	Alliance	Priest	0	2	4	9462	6233	153	\N	1	dps	\N
4132	TK	TK28	Alliance	Hunter	0	3	2	8125	24	146	\N	1	dps	\N
4133	TK	TK28	Horde	Shaman	1	1	32	7405	38314	370	1	\N	heal	\N
4134	TK	TK28	Horde	Mage	4	0	36	37396	3482	374	1	\N	dps	\N
4135	TK	TK28	Horde	Hunter	3	0	36	59535	1452	378	1	\N	dps	\N
4136	TK	TK28	Alliance	Druid	0	4	4	7485	3762	153	\N	1	dps	\N
4137	TK	TK28	Horde	Paladin	3	1	30	17903	11241	516	1	\N	dps	\N
4138	TK	TK28	Alliance	Mage	0	4	4	10755	7558	153	\N	1	dps	\N
4139	TK	TK28	Alliance	Warlock	0	4	3	18582	3782	149	\N	1	dps	\N
4140	TK	TK28	Alliance	Rogue	0	4	4	22115	0	153	\N	1	dps	\N
4141	TK	TK28	Alliance	Warrior	1	5	4	50276	1613	153	\N	1	dps	\N
4142	TK	TK28	Horde	Shaman	0	0	36	4813	70012	378	1	\N	heal	\N
4143	TK	TK29	Alliance	Paladin	2	2	14	38821	19496	236	\N	1	dps	\N
4144	TK	TK29	Horde	Warrior	4	3	32	31925	4893	518	1	\N	dps	\N
4145	TK	TK29	Alliance	Hunter	2	4	16	31296	3807	238	\N	1	dps	\N
4146	TK	TK29	Alliance	Mage	0	3	18	30155	8495	244	\N	1	dps	\N
4147	TK	TK29	Alliance	Warrior	1	6	14	33514	2108	233	\N	1	dps	\N
4148	TK	TK29	Horde	Rogue	4	1	32	36120	6815	368	1	\N	dps	\N
4149	TK	TK29	Horde	Shaman	0	2	33	13806	98538	520	1	\N	heal	\N
4150	TK	TK29	Horde	Shaman	0	3	28	1441	70523	360	1	\N	heal	\N
4151	TK	TK29	Horde	Warrior	6	2	28	59682	4309	510	1	\N	dps	\N
4152	TK	TK29	Alliance	Mage	6	5	16	96631	4271	238	\N	1	dps	\N
4153	TK	TK29	Alliance	Druid	0	3	13	5753	72375	230	\N	1	heal	\N
4154	TK	TK29	Horde	Rogue	4	0	33	18836	10123	510	1	\N	dps	\N
4155	TK	TK29	Alliance	Hunter	2	4	18	25417	7118	244	\N	1	dps	\N
4156	TK	TK29	Alliance	Druid	0	1	18	7369	6601	244	\N	1	dps	\N
4157	TK	TK29	Horde	Druid	3	2	32	57571	6090	518	1	\N	dps	\N
4158	TK	TK29	Horde	Rogue	3	1	34	53048	6765	372	1	\N	dps	\N
4159	TK	TK29	Horde	Death Knight	1	3	33	32887	24753	520	1	\N	dps	\N
4160	TK	TK29	Horde	Demon Hunter	8	1	34	64269	3439	372	1	\N	dps	\N
4161	TK	TK29	Alliance	Priest	0	4	14	7573	56754	232	\N	1	heal	\N
4162	TK	TK30	Horde	Warrior	15	2	68	66185	5171	435	1	\N	dps	\N
4163	TK	TK30	Horde	Druid	0	1	64	2163	174000	427	1	\N	heal	\N
4164	TK	TK30	Horde	Hunter	4	4	60	67632	13870	418	1	\N	dps	\N
4165	TK	TK30	Alliance	Warlock	2	5	29	61679	38866	417	\N	1	dps	\N
4166	TK	TK30	Alliance	Rogue	7	5	29	69788	12341	417	\N	1	dps	\N
4167	TK	TK30	Alliance	Hunter	4	6	27	68495	23956	411	\N	1	dps	\N
4168	TK	TK30	Alliance	Warrior	0	9	22	35093	5991	394	\N	1	dps	\N
4169	TK	TK30	Horde	Warlock	4	3	59	43720	9135	566	1	\N	dps	\N
4170	TK	TK30	Horde	Shaman	0	6	55	7508	95265	408	1	\N	heal	\N
4171	TK	TK30	Horde	Demon Hunter	8	4	56	92223	24163	411	1	\N	dps	\N
4172	TK	TK30	Alliance	Warlock	4	7	27	47053	20279	409	\N	1	dps	\N
4173	TK	TK30	Alliance	Hunter	1	8	24	41114	14760	398	\N	1	dps	\N
4174	TK	TK30	Alliance	Warlock	7	5	31	93291	51750	421	\N	1	dps	\N
4175	TK	TK30	Horde	Death Knight	10	4	61	110000	29781	571	1	\N	dps	\N
4176	TK	TK30	Horde	Death Knight	10	6	64	79357	39566	275	1	\N	dps	\N
4177	TK	TK30	Alliance	Mage	9	6	28	129000	17565	413	\N	1	dps	\N
4178	TK	TK30	Alliance	Shaman	0	7	27	4126	53718	409	\N	1	heal	\N
4179	TK	TK30	Alliance	Priest	1	9	27	49829	19258	410	\N	1	dps	\N
4180	TK	TK30	Horde	Mage	9	2	62	93060	17061	572	1	\N	dps	\N
4181	TK	TK30	Horde	Demon Hunter	7	3	58	69499	5511	414	1	\N	dps	\N
4182	TK	TK31	Horde	Shaman	2	6	33	24152	4962	264	\N	1	dps	\N
4183	TK	TK31	Horde	Rogue	6	5	34	40831	10648	268	\N	1	dps	\N
4184	TK	TK31	Alliance	Paladin	4	3	59	37246	12544	612	1	\N	dps	\N
4185	TK	TK31	Horde	Death Knight	4	5	36	57082	28817	272	\N	1	dps	\N
4186	TK	TK31	Alliance	Demon Hunter	3	5	58	48130	4521	844	1	\N	dps	\N
4187	TK	TK31	Alliance	Warrior	10	4	57	105000	27371	612	1	\N	dps	\N
4188	TK	TK31	Alliance	Mage	0	8	39	24957	8796	799	1	\N	dps	\N
4189	TK	TK31	Horde	Priest	5	5	40	56318	26732	281	\N	1	dps	\N
4190	TK	TK31	Alliance	Mage	1	5	59	45045	6296	619	1	\N	dps	\N
4191	TK	TK31	Horde	Shaman	0	9	28	25938	84548	254	\N	1	heal	\N
4192	TK	TK31	Alliance	Death Knight	3	4	57	70136	26207	614	1	\N	dps	\N
4193	TK	TK31	Horde	Rogue	5	4	42	53193	16934	285	\N	1	dps	\N
4194	TK	TK31	Horde	Warlock	6	7	36	99579	38579	271	\N	1	dps	\N
4195	TK	TK31	Horde	Shaman	5	9	33	58546	22037	264	\N	1	dps	\N
4196	TK	TK31	Horde	Priest	2	6	38	5902	110000	277	\N	1	heal	\N
4197	TK	TK31	Alliance	Warlock	7	3	61	87281	52407	615	1	\N	dps	\N
4198	TK	TK31	Alliance	Druid	4	4	57	72397	23753	616	1	\N	dps	\N
4199	TK	TK31	Alliance	Warrior	20	3	59	84419	31867	613	1	\N	dps	\N
4200	TK	TK31	Horde	Druid	7	7	37	63078	10799	273	\N	1	dps	\N
4201	TK	TK31	Alliance	Death Knight	10	4	53	73122	29725	589	1	\N	dps	\N
4202	TK	TK32	Alliance	Warrior	8	2	55	89690	11771	595	1	\N	dps	\N
4203	TK	TK32	Horde	Mage	3	5	13	38565	11677	200	\N	1	dps	\N
4204	TK	TK32	Alliance	Warrior	18	2	51	125000	22416	807	1	\N	dps	\N
4205	TK	TK32	Alliance	Warrior	1	5	48	10258	4514	578	1	\N	dps	\N
4206	TK	TK32	Horde	Priest	0	4	17	3469	58348	208	\N	1	heal	\N
4207	TK	TK32	Alliance	Demon Hunter	3	1	55	47394	4342	595	1	\N	dps	\N
4208	TK	TK32	Horde	Druid	1	6	14	27170	10446	202	\N	1	dps	\N
4209	TK	TK32	Alliance	Paladin	11	2	52	85531	21211	588	1	\N	dps	\N
4210	TK	TK32	Alliance	Warrior	9	1	53	35697	2991	586	1	\N	dps	\N
4211	TK	TK32	Alliance	Priest	5	1	55	94639	6911	817	1	\N	dps	\N
4212	TK	TK32	Alliance	Warrior	0	4	43	24060	1961	793	1	\N	dps	\N
4213	TK	TK32	Alliance	Paladin	0	0	55	2222	84019	595	1	\N	heal	\N
4214	TK	TK32	Horde	Shaman	0	6	14	15020	63435	202	\N	1	heal	\N
4215	TK	TK32	Horde	Mage	1	3	13	23441	3283	200	\N	1	dps	\N
4216	TK	TK32	Horde	Warlock	4	6	13	52498	20817	200	\N	1	dps	\N
4217	TK	TK32	Alliance	Monk	0	0	55	378	151000	595	1	\N	heal	\N
4218	TK	TK32	Horde	Death Knight	1	7	13	38963	17063	200	\N	1	dps	\N
4219	TK	TK32	Horde	Druid	0	1	3	185	12731	111	\N	1	heal	\N
4220	TK	TK32	Horde	Hunter	2	7	13	102000	98012	200	\N	1	dps	\N
4221	TK	TK32	Horde	Hunter	1	7	14	74785	9045	202	\N	1	dps	\N
4222	TP	TP20	Horde	Mage	2	4	26	49446	10841	193	\N	1	dps	\N
4223	TP	TP20	Horde	Priest	0	3	26	8487	79988	194	\N	1	heal	\N
4224	TP	TP20	Alliance	Warlock	7	2	41	161000	84231	829	1	\N	dps	\N
4225	TP	TP20	Alliance	Shaman	0	3	27	7059	62691	789	1	\N	heal	\N
4226	TP	TP20	Alliance	Warrior	5	0	36	53887	17832	827	1	\N	dps	\N
4227	TP	TP20	Horde	Hunter	4	5	22	90660	11438	190	\N	1	dps	\N
4228	TP	TP20	Horde	Hunter	5	6	30	69237	5640	202	\N	1	dps	\N
4229	TP	TP20	Horde	Shaman	0	7	21	26976	150000	183	\N	1	heal	\N
4230	TP	TP20	Horde	Priest	4	6	24	30687	13733	195	\N	1	dps	\N
4231	TP	TP20	Alliance	Paladin	0	3	29	6462	105000	787	1	\N	heal	\N
4232	TP	TP20	Horde	Shaman	0	4	27	10800	146000	197	\N	1	heal	\N
4233	TP	TP20	Horde	Death Knight	7	5	27	161000	28546	195	\N	1	dps	\N
4234	TP	TP20	Horde	Shaman	4	5	29	47666	10671	201	\N	1	dps	\N
4235	TP	TP20	Alliance	Warrior	11	5	38	75466	26947	589	1	\N	dps	\N
4236	TP	TP20	Alliance	Priest	8	3	35	96725	17765	815	1	\N	dps	\N
4237	TP	TP20	Alliance	Death Knight	6	3	38	115000	29834	589	1	\N	dps	\N
4238	TP	TP20	Alliance	Warrior	5	6	33	74281	15159	798	1	\N	dps	\N
4239	TP	TP20	Alliance	Warlock	6	3	40	97285	20120	820	1	\N	dps	\N
4240	TP	TP20	Alliance	Warlock	4	5	32	47222	29288	571	1	\N	dps	\N
4241	TP	TP21	Alliance	Priest	2	1	24	23986	11571	597	1	\N	dps	1
4242	TP	TP21	Alliance	Mage	12	3	31	123000	16627	613	1	\N	dps	1
4243	TP	TP21	Horde	Death Knight	1	4	25	58146	17166	232	\N	1	dps	1
4244	TP	TP21	Horde	Priest	4	4	24	44636	4602	229	\N	1	dps	1
4245	TP	TP21	Horde	Priest	1	5	23	9148	90863	225	\N	1	heal	1
4246	TP	TP21	Horde	Shaman	1	4	16	71623	15668	177	\N	1	dps	1
4247	TP	TP21	Horde	Shaman	1	1	19	17405	45947	206	\N	1	heal	1
4248	TP	TP21	Horde	Warrior	9	4	27	70786	10526	239	\N	1	dps	1
4249	TP	TP21	Horde	Shaman	5	4	20	99743	15667	217	\N	1	dps	1
4250	TP	TP21	Alliance	Rogue	2	4	27	65567	14002	604	1	\N	dps	1
4251	TP	TP21	Horde	Priest	8	3	19	164000	35874	214	\N	1	dps	1
4252	TP	TP21	Horde	Shaman	0	0	9	8544	13642	145	\N	1	heal	1
4253	TP	TP21	Alliance	Mage	6	3	31	80915	8821	613	1	\N	dps	1
4254	TP	TP21	Alliance	Druid	0	1	21	2824	91371	246	1	\N	heal	1
4255	TP	TP21	Horde	Priest	0	3	19	24276	104000	214	\N	1	heal	1
4256	TP	TP21	Alliance	Mage	0	6	30	29125	7659	947	1	\N	dps	1
4257	TP	TP21	Alliance	Mage	4	5	17	51172	12978	594	1	\N	dps	1
4258	TP	TP21	Alliance	Monk	0	2	31	8832	188000	613	1	\N	heal	1
4259	TP	TP21	Alliance	Priest	6	4	31	134000	41314	613	1	\N	dps	1
4260	TP	TP21	Alliance	Druid	1	3	21	30263	23336	920	1	\N	dps	1
4261	TP	TP22	Horde	Priest	3	0	27	25521	7568	363	1	\N	dps	\N
4262	TP	TP22	Alliance	Mage	1	4	8	32625	16571	181	\N	1	dps	\N
4263	TP	TP22	Horde	Death Knight	8	2	34	49275	22545	531	1	\N	dps	\N
4264	TP	TP22	Horde	Warlock	1	1	34	32927	13968	380	1	\N	dps	\N
4265	TP	TP22	Alliance	Demon Hunter	1	6	6	30388	4327	173	\N	1	dps	\N
4266	TP	TP22	Alliance	Mage	1	2	8	22873	10859	181	\N	1	dps	\N
4267	TP	TP22	Horde	Death Knight	4	3	33	40555	30877	529	1	\N	dps	\N
4268	TP	TP22	Horde	Shaman	0	0	29	10375	40682	517	1	\N	heal	\N
4269	TP	TP22	Horde	Priest	3	1	29	14032	4485	367	1	\N	dps	\N
4270	TP	TP22	Alliance	Shaman	0	3	3	7211	113712	165	\N	1	heal	\N
4271	TP	TP22	Horde	Mage	6	0	34	27751	10156	379	1	\N	dps	\N
4272	TP	TP22	Horde	Paladin	7	1	34	60236	17928	380	1	\N	dps	\N
4273	TP	TP22	Alliance	Hunter	0	3	5	13082	1432	156	\N	1	dps	\N
4274	TP	TP22	Alliance	Shaman	4	4	9	43926	15883	184	\N	1	dps	\N
4275	TP	TP22	Alliance	Shaman	0	5	3	1041	15967	164	\N	1	heal	\N
4276	TP	TP22	Alliance	Mage	1	2	7	20456	3081	164	\N	1	dps	\N
4277	TP	TP22	Alliance	Priest	0	2	7	4860	5683	162	\N	1	heal	\N
4278	TP	TP22	Horde	Warlock	4	1	29	47244	16348	520	1	\N	dps	\N
4279	TP	TP22	Horde	Mage	1	0	25	3943	5309	361	1	\N	dps	\N
4280	TP	TP23	Horde	Druid	1	0	18	3190	45142	515	1	\N	heal	\N
4281	TP	TP23	Alliance	Druid	0	3	7	4010	12027	190	\N	1	heal	\N
4282	TP	TP23	Horde	Shaman	0	2	26	1187	43768	532	1	\N	heal	\N
4283	TP	TP23	Alliance	Demon Hunter	0	0	7	21752	5291	144	\N	1	dps	\N
4284	TP	TP23	Alliance	Priest	3	0	8	11116	2438	162	\N	1	dps	\N
4285	TP	TP23	Horde	Paladin	6	3	28	91108	24208	543	1	\N	dps	\N
4286	TP	TP23	Alliance	Mage	1	3	12	20240	6525	206	\N	1	dps	\N
4287	TP	TP23	Horde	Paladin	1	0	24	59417	16324	532	1	\N	dps	\N
4288	TP	TP23	Alliance	Mage	2	2	13	54098	19283	210	\N	1	dps	\N
4289	TP	TP23	Alliance	Warrior	2	0	7	5038	709	144	\N	1	dps	\N
4290	TP	TP23	Horde	Priest	6	2	26	69323	20374	385	1	\N	dps	\N
4291	TP	TP23	Horde	Death Knight	5	3	29	59033	26121	392	1	\N	dps	\N
4292	TP	TP23	Alliance	Warrior	2	7	14	53234	13878	213	\N	1	dps	\N
4293	TP	TP23	Horde	Rogue	4	1	25	26222	5849	533	1	\N	dps	\N
4294	TP	TP23	Horde	Death Knight	1	0	22	31064	15050	373	1	\N	dps	\N
4295	TP	TP23	Alliance	Warlock	1	4	9	57661	30847	195	\N	1	dps	\N
4296	TP	TP23	Horde	Priest	0	1	13	28681	5071	355	1	\N	dps	\N
4297	TP	TP23	Alliance	Paladin	0	5	10	2644	97761	198	\N	1	heal	\N
4298	TP	TP23	Alliance	Death Knight	1	5	7	36616	6457	193	\N	1	dps	\N
4299	TP	TP23	Horde	Warlock	7	2	30	32919	12861	393	1	\N	dps	\N
4300	WG	WG36	Alliance	Druid	0	0	21	8116	100000	665	1	\N	heal	1
4301	WG	WG36	Alliance	Warrior	4	1	16	58678	14506	646	1	\N	dps	1
4302	WG	WG36	Horde	Priest	0	0	0	5516	8418	122	\N	1	heal	1
4303	WG	WG36	Alliance	Priest	8	0	21	70103	18631	1002	1	\N	dps	1
4304	WG	WG36	Horde	Mage	1	1	12	56959	5366	174	\N	1	dps	1
4305	WG	WG36	Horde	Druid	0	2	8	34732	13235	166	\N	1	dps	1
4306	WG	WG36	Horde	Shaman	0	4	8	11985	55476	166	\N	1	heal	1
4307	WG	WG36	Horde	Demon Hunter	2	2	10	53349	12465	170	\N	1	dps	1
4308	WG	WG36	Alliance	Warrior	1	4	17	41935	9694	645	1	\N	dps	1
4309	WG	WG36	Horde	Warrior	3	2	9	32665	4082	46	\N	1	dps	1
4310	WG	WG36	Alliance	Hunter	0	2	10	15242	4028	631	1	\N	dps	1
4311	WG	WG36	Alliance	Paladin	2	0	18	85703	10796	658	1	\N	dps	1
4312	WG	WG36	Alliance	Warlock	6	1	18	54080	11981	651	1	\N	dps	1
4313	WG	WG36	Horde	Rogue	0	2	10	43950	13023	170	\N	1	dps	1
4314	WG	WG36	Alliance	Druid	0	2	18	2429	117000	648	1	\N	heal	1
4315	WG	WG36	Alliance	Death Knight	1	2	18	88435	30029	650	1	\N	dps	1
4316	WG	WG36	Horde	Priest	0	0	2	21265	43585	136	\N	1	heal	1
4317	WG	WG36	Horde	Monk	0	2	10	1516	143000	170	\N	1	heal	1
4318	WG	WG36	Alliance	Warrior	0	0	17	5751	6460	647	1	\N	dps	1
4319	WG	WG36	Horde	Mage	1	1	11	82577	6493	172	\N	1	dps	1
4320	WG	WG37	Horde	Mage	4	2	21	60439	11101	543	1	\N	dps	\N
4321	WG	WG37	Alliance	Priest	0	3	12	6088	86037	236	\N	1	heal	\N
4322	WG	WG37	Alliance	Warrior	1	4	15	27380	25164	245	\N	1	dps	\N
4323	WG	WG37	Alliance	Paladin	0	2	14	21169	28302	131	\N	1	heal	\N
4324	WG	WG37	Alliance	Hunter	3	3	19	82268	11431	259	\N	1	dps	\N
4325	WG	WG37	Alliance	Rogue	5	4	17	58305	7070	140	\N	1	dps	\N
4326	WG	WG37	Alliance	Death Knight	0	1	0	6812	1500	127	\N	1	dps	\N
4327	WG	WG37	Alliance	Demon Hunter	0	7	13	66609	9221	237	\N	1	dps	\N
4328	WG	WG37	Horde	Death Knight	0	3	30	37442	10666	266	1	\N	dps	\N
4329	WG	WG37	Horde	Priest	0	3	23	13121	55945	552	1	\N	heal	\N
4330	WG	WG37	Horde	Paladin	3	0	27	33544	83773	553	1	\N	heal	\N
4331	WG	WG37	Horde	Shaman	1	0	26	23963	76586	551	1	\N	heal	\N
4332	WG	WG37	Horde	Hunter	0	6	31	35227	8590	418	1	\N	dps	\N
4333	WG	WG37	Alliance	Hunter	1	3	13	23972	1098	128	\N	1	dps	\N
4334	WG	WG37	Horde	Monk	5	0	26	59114	27812	402	1	\N	dps	\N
4335	WG	WG37	Horde	Druid	4	0	25	25783	19455	249	1	\N	dps	\N
4336	WG	WG37	Horde	Warrior	14	4	30	83696	20219	412	1	\N	dps	\N
4337	WG	WG37	Alliance	Warrior	7	5	17	53254	7422	252	\N	1	dps	\N
4338	WG	WG37	Alliance	Hunter	0	1	3	12118	1445	137	\N	1	dps	\N
4339	WG	WG37	Horde	Warlock	10	2	33	104000	25535	568	1	\N	dps	\N
4340	WG	WG38	Horde	Paladin	0	5	16	29372	13745	170	\N	1	dps	\N
4341	WG	WG38	Alliance	Paladin	1	2	47	8059	175000	536	1	\N	heal	\N
4342	WG	WG38	Alliance	Hunter	5	4	44	69902	8554	754	1	\N	dps	\N
4343	WG	WG38	Horde	Rogue	3	1	19	27338	9139	101	\N	1	dps	\N
4344	WG	WG38	Alliance	Rogue	7	1	34	50432	4370	720	1	\N	dps	\N
4345	WG	WG38	Alliance	Warlock	0	0	18	16331	2870	357	1	\N	dps	\N
4346	WG	WG38	Alliance	Demon Hunter	5	3	39	77434	12626	513	1	\N	dps	\N
4347	WG	WG38	Alliance	Paladin	4	1	36	57167	12416	743	1	\N	dps	\N
4348	WG	WG38	Horde	Shaman	0	6	17	21569	133000	172	\N	1	heal	\N
4349	WG	WG38	Horde	Priest	0	7	12	2765	97789	162	\N	1	heal	\N
4350	WG	WG38	Alliance	Warlock	7	5	46	132000	36541	527	1	\N	dps	\N
4351	WG	WG38	Horde	Warrior	4	2	11	62804	7333	120	\N	1	dps	\N
4352	WG	WG38	Alliance	Monk	2	0	48	7117	285000	763	1	\N	heal	\N
4353	WG	WG38	Alliance	Paladin	3	1	42	46873	8412	732	1	\N	dps	\N
4354	WG	WG38	Horde	Warlock	4	5	19	137000	87026	176	\N	1	dps	\N
4355	WG	WG38	Horde	Shaman	0	1	11	8197	37806	120	\N	1	heal	\N
4356	WG	WG38	Alliance	Hunter	3	2	49	178000	7388	542	1	\N	dps	\N
4357	WG	WG38	Horde	Warlock	3	2	18	152000	74004	174	\N	1	dps	\N
4358	WG	WG38	Horde	Druid	1	8	16	94675	24877	170	\N	1	dps	\N
4359	WG	WG38	Horde	Hunter	3	4	16	84204	17573	170	\N	1	dps	\N
4360	WG	WG39	Alliance	Priest	5	6	14	66183	21087	291	\N	1	dps	\N
4361	WG	WG39	Horde	Paladin	6	1	49	86093	37884	420	1	\N	dps	\N
4362	WG	WG39	Alliance	Paladin	0	4	14	1860	99695	289	\N	1	heal	\N
4363	WG	WG39	Horde	Shaman	1	2	39	12656	93962	400	1	\N	heal	\N
4364	WG	WG39	Horde	Warlock	6	3	56	96248	56075	587	1	\N	dps	\N
4365	WG	WG39	Horde	Druid	6	1	55	87299	9646	584	1	\N	dps	\N
4366	WG	WG39	Horde	Shaman	8	3	45	46703	12549	415	1	\N	dps	\N
4367	WG	WG39	Horde	Demon Hunter	2	2	52	59141	8782	579	1	\N	dps	\N
4368	WG	WG39	Horde	Shaman	1	3	41	13261	61877	554	1	\N	heal	\N
4369	WG	WG39	Alliance	Warlock	0	2	3	18346	11932	135	\N	1	dps	\N
4370	WG	WG39	Alliance	Shaman	0	5	14	42479	23241	291	\N	1	dps	\N
4371	WG	WG39	Horde	Shaman	9	4	55	72572	20143	517	1	\N	dps	\N
4372	WG	WG39	Alliance	Mage	2	5	14	55189	12165	290	\N	1	dps	\N
4373	WG	WG39	Alliance	Druid	2	8	16	69180	7850	295	\N	1	dps	\N
4374	WG	WG39	Alliance	Paladin	6	6	16	94705	24637	296	\N	1	dps	\N
4375	WG	WG39	Horde	Hunter	6	1	51	80365	5586	575	1	\N	dps	\N
4376	WG	WG39	Alliance	Priest	2	6	18	30886	166000	302	\N	1	heal	\N
4377	WG	WG39	Horde	Priest	10	2	58	117000	28359	588	1	\N	dps	\N
4378	WG	WG39	Alliance	Rogue	0	5	17	13890	13005	298	\N	1	dps	\N
4379	WG	WG39	Alliance	Warrior	2	7	17	26381	3314	297	\N	1	dps	\N
4380	AB	AB14	Horde	Demon Hunter	6	1	60	70977	20655	570	1	\N	dps	\N
4381	AB	AB14	Alliance	Priest	2	7	34	33024	151000	413	\N	1	heal	\N
4382	AB	AB14	Alliance	Paladin	6	4	40	119000	42079	443	\N	1	dps	\N
4383	AB	AB14	Horde	Paladin	2	5	55	53704	23893	410	1	\N	dps	\N
4384	AB	AB14	Horde	Druid	1	5	55	55452	22233	546	1	\N	dps	\N
4385	AB	AB14	Horde	Paladin	8	4	58	119000	37772	425	1	\N	dps	\N
4386	AB	AB14	Alliance	Mage	4	3	32	73432	12369	402	\N	1	dps	\N
4387	AB	AB14	Alliance	Druid	3	6	30	72133	15189	416	\N	1	dps	\N
4388	AB	AB14	Horde	Mage	6	6	62	66488	22601	581	1	\N	dps	\N
4389	AB	AB14	Horde	Monk	1	1	73	18766	235000	469	1	\N	heal	\N
4390	AB	AB14	Alliance	Warrior	7	9	34	69436	19431	422	\N	1	dps	\N
4391	AB	AB14	Alliance	Demon Hunter	4	4	22	66654	8177	386	\N	1	dps	\N
4392	AB	AB14	Horde	Shaman	1	8	51	24738	167000	563	1	\N	heal	\N
4393	AB	AB14	Horde	Hunter	7	2	63	84028	9078	589	1	\N	dps	\N
4394	AB	AB14	Alliance	Hunter	1	6	34	80095	26533	413	\N	1	dps	\N
4395	AB	AB14	Alliance	Hunter	1	0	32	41387	16359	417	\N	1	dps	\N
4396	AB	AB14	Horde	Rogue	4	4	58	80608	13360	571	1	\N	dps	\N
4397	AB	AB14	Alliance	Shaman	1	2	37	5365	152000	419	\N	1	heal	\N
4398	AB	AB14	Alliance	Rogue	3	7	30	32258	14873	409	\N	1	dps	\N
4399	AB	AB14	Horde	Priest	8	0	63	102000	41460	424	1	\N	dps	\N
4400	AB	AB14	Alliance	Paladin	8	7	32	112000	29352	428	\N	1	dps	\N
4401	AB	AB14	Alliance	Warlock	7	6	43	202000	92901	453	\N	1	dps	\N
4402	AB	AB14	Alliance	Warrior	8	8	34	62199	13338	414	\N	1	dps	\N
4403	AB	AB14	Horde	Warlock	9	5	56	87740	37946	418	1	\N	dps	\N
4404	AB	AB14	Horde	Mage	6	6	64	116000	21981	424	1	\N	dps	\N
4405	AB	AB14	Horde	Shaman	1	3	20	38393	18649	438	1	\N	dps	\N
4406	AB	AB14	Alliance	Hunter	3	7	30	58522	13855	416	\N	1	dps	\N
4407	AB	AB14	Horde	Demon Hunter	4	6	41	76334	11831	546	1	\N	dps	\N
4408	AB	AB14	Horde	Warrior	13	4	66	82950	29093	431	1	\N	dps	\N
4409	AB	AB14	Alliance	Druid	3	7	26	65198	22194	394	\N	1	dps	\N
4410	AB	AB15	Alliance	Druid	6	4	12	33564	15275	259	\N	1	dps	\N
4411	AB	AB15	Alliance	Priest	1	7	12	12021	52926	248	\N	1	heal	\N
4412	AB	AB15	Alliance	Monk	2	8	18	29782	11293	273	\N	1	dps	\N
4413	AB	AB15	Horde	Druid	2	0	39	41785	11272	532	1	\N	dps	\N
4414	AB	AB15	Horde	Demon Hunter	13	2	54	70287	21112	567	1	\N	dps	\N
4415	AB	AB15	Alliance	Shaman	0	1	16	46972	14611	264	\N	1	dps	\N
4416	AB	AB15	Alliance	Warrior	4	6	10	50725	9232	248	\N	1	dps	\N
4417	AB	AB15	Horde	Shaman	1	0	49	24472	110000	410	1	\N	heal	\N
4418	AB	AB15	Horde	Warlock	5	1	47	41913	22924	419	1	\N	dps	\N
4419	AB	AB15	Alliance	Warrior	5	6	18	67996	16946	271	\N	1	dps	\N
4420	AB	AB15	Horde	Priest	6	1	28	34742	10611	524	1	\N	dps	\N
4421	AB	AB15	Horde	Monk	6	3	61	75674	30276	566	1	\N	dps	\N
4422	AB	AB15	Alliance	Paladin	0	9	18	37704	12009	271	\N	1	dps	\N
4423	AB	AB15	Alliance	Warrior	2	9	12	69180	12948	248	\N	1	dps	\N
4424	AB	AB15	Alliance	Druid	0	1	5	0	49057	230	\N	1	heal	\N
4425	AB	AB15	Horde	Druid	0	0	38	453	92823	542	1	\N	heal	\N
4426	AB	AB15	Horde	Rogue	4	2	58	41820	20465	423	1	\N	dps	\N
4427	AB	AB15	Horde	Rogue	2	4	55	24559	7891	597	1	\N	dps	\N
4428	AB	AB15	Horde	Death Knight	12	2	59	95063	20990	575	1	\N	dps	\N
4429	AB	AB15	Horde	Priest	2	1	64	52224	213000	421	1	\N	heal	\N
4430	AB	AB15	Horde	Warlock	9	3	55	47945	27165	412	1	\N	dps	\N
4431	AB	AB15	Alliance	Hunter	3	2	9	52481	9195	246	\N	1	dps	\N
4432	AB	AB15	Horde	Paladin	3	0	39	70628	18684	542	1	\N	dps	\N
4433	AB	AB15	Alliance	Paladin	1	4	7	31422	18428	235	\N	1	dps	\N
4434	AB	AB15	Alliance	Druid	2	7	15	52201	1899	263	\N	1	dps	\N
4435	AB	AB15	Alliance	Priest	1	6	16	89347	22629	261	\N	1	dps	\N
4436	AB	AB15	Alliance	Demon Hunter	1	6	12	134000	8068	252	\N	1	dps	\N
4437	AB	AB15	Horde	Hunter	10	1	65	82505	6979	573	1	\N	dps	\N
4438	AB	AB15	Horde	Death Knight	5	4	58	52951	15687	419	1	\N	dps	\N
4439	AB	AB15	Alliance	Hunter	1	4	6	15642	2172	201	\N	1	dps	\N
4440	AB	AB16	Alliance	Death Knight	2	6	26	90304	17416	371	\N	1	dps	1
4441	AB	AB16	Horde	Hunter	2	1	49	79915	8839	488	1	\N	dps	1
4442	AB	AB16	Alliance	Death Knight	0	4	27	18363	9810	382	\N	1	dps	1
4443	AB	AB16	Alliance	Warrior	9	4	27	80245	10083	374	\N	1	dps	1
4444	AB	AB16	Horde	Shaman	11	1	52	128000	25531	701	1	\N	dps	1
4445	AB	AB16	Alliance	Shaman	3	4	23	87413	26764	387	\N	1	dps	1
4446	AB	AB16	Horde	Rogue	7	6	30	65289	13319	456	1	\N	dps	1
4447	AB	AB16	Horde	Rogue	2	0	21	8531	70	445	1	\N	dps	1
4448	AB	AB16	Alliance	Mage	5	7	21	113000	24819	361	\N	1	dps	1
4449	AB	AB16	Alliance	Death Knight	2	2	15	32026	14514	349	\N	1	dps	1
4450	AB	AB16	Alliance	Paladin	2	3	20	91332	29720	374	\N	1	dps	1
4451	AB	AB16	Horde	Death Knight	4	3	46	59947	68447	695	1	\N	dps	1
4452	AB	AB16	Horde	Demon Hunter	3	4	33	86990	9239	461	1	\N	dps	1
4453	AB	AB16	Horde	Shaman	0	4	48	22986	130000	471	1	\N	heal	1
4454	AB	AB16	Alliance	Priest	0	6	30	33507	147000	383	\N	1	heal	1
4455	AB	AB16	Alliance	Rogue	3	5	28	52354	6833	385	\N	1	dps	1
4456	AB	AB16	Horde	Shaman	0	2	55	9630	156000	721	1	\N	heal	1
4457	AB	AB16	Alliance	Demon Hunter	0	7	16	107000	17878	345	\N	1	dps	1
4458	AB	AB16	Horde	Warrior	18	2	53	134000	17062	509	1	\N	dps	1
4459	AB	AB16	Horde	Demon Hunter	4	1	28	64683	31672	455	1	\N	dps	1
4460	AB	AB16	Horde	Priest	0	1	56	20190	210000	502	1	\N	heal	1
4461	AB	AB16	Alliance	Death Knight	1	5	20	65751	17930	184	\N	1	dps	1
4462	AB	AB16	Alliance	Warrior	8	4	18	63829	16746	353	\N	1	dps	1
4463	AB	AB16	Alliance	Paladin	1	7	19	69818	16741	186	\N	1	dps	1
4464	AB	AB16	Horde	Druid	4	4	47	77643	23719	482	1	\N	dps	1
4465	AB	AB16	Horde	Rogue	6	1	45	63865	8363	701	1	\N	dps	1
4466	AB	AB16	Horde	Rogue	1	6	31	35819	10836	677	1	\N	dps	1
4467	AB	AB16	Alliance	Druid	0	4	14	4993	110000	347	\N	1	heal	1
4468	AB	AB16	Horde	Mage	4	5	37	92289	12878	680	1	\N	dps	1
4469	AB	AB16	Alliance	Monk	0	4	25	2707	96889	372	\N	1	heal	1
4470	SA	SA5	Horde	Priest	1	2	34	27692	222	354	1	\N	dps	\N
4471	SA	SA5	Horde	Hunter	1	6	28	49060	8143	192	1	\N	dps	\N
4472	SA	SA5	Alliance	Rogue	3	1	42	54417	6343	335	\N	1	dps	\N
4473	SA	SA5	Horde	Mage	0	0	36	150000	10649	358	1	\N	dps	\N
4474	SA	SA5	Alliance	Priest	7	2	38	184000	21679	318	\N	1	dps	\N
4475	SA	SA5	Horde	Rogue	3	4	34	86417	6003	349	1	\N	dps	\N
4476	SA	SA5	Alliance	Hunter	5	2	36	84526	7123	325	\N	1	dps	\N
4477	SA	SA5	Horde	Rogue	1	4	36	92206	12729	506	1	\N	dps	\N
4478	SA	SA5	Alliance	Mage	8	1	43	148000	20298	342	\N	1	dps	\N
4479	SA	SA5	Horde	Warrior	3	9	30	116000	19390	348	1	\N	dps	\N
4480	SA	SA5	Alliance	Warrior	2	5	32	69929	2970	309	\N	1	dps	\N
4481	SA	SA5	Horde	Hunter	3	0	29	53204	9763	198	1	\N	dps	\N
4482	SA	SA5	Horde	Shaman	0	0	33	21668	184000	503	1	\N	heal	\N
4483	SA	SA5	Horde	Warrior	1	3	32	58086	5265	492	1	\N	dps	\N
4484	SA	SA5	Horde	Warlock	5	4	39	108000	38592	361	1	\N	dps	\N
4485	SA	SA5	Horde	Shaman	9	1	39	71097	26900	498	1	\N	dps	\N
4486	SA	SA5	Alliance	Paladin	1	0	27	18612	48750	240	\N	1	heal	\N
4487	SA	SA5	Alliance	Warlock	5	2	45	90654	12704	351	\N	1	dps	\N
4488	SA	SA5	Alliance	Priest	4	2	42	40277	127000	343	\N	1	heal	\N
4489	SA	SA5	Alliance	Priest	0	5	37	18131	125000	325	\N	1	heal	\N
4490	SA	SA5	Horde	Monk	3	6	35	57983	34554	356	1	\N	dps	\N
4491	SA	SA5	Alliance	Druid	0	2	44	13613	144000	348	\N	1	heal	\N
4492	SA	SA5	Horde	Priest	0	1	27	42151	171000	338	1	\N	heal	\N
4493	SA	SA5	Horde	Demon Hunter	4	7	30	98092	20535	349	1	\N	dps	\N
4494	SA	SA5	Horde	Paladin	6	4	38	136000	52794	357	1	\N	dps	\N
4495	SA	SA5	Alliance	Death Knight	6	1	32	153000	43691	325	\N	1	dps	\N
4496	SA	SA5	Alliance	Druid	4	6	34	148000	1949	311	\N	1	dps	\N
4497	SA	SA5	Alliance	Mage	4	5	35	76572	7607	300	\N	1	dps	\N
4498	SA	SA5	Alliance	Monk	0	4	35	6880	236000	319	\N	1	heal	\N
4499	SA	SA5	Alliance	Hunter	0	3	33	22401	3162	205	\N	1	dps	\N
4500	SA	SA6	Alliance	Shaman	0	7	61	6717	127000	703	1	\N	heal	1
4501	SA	SA6	Horde	Druid	3	5	23	65371	25229	236	\N	1	dps	1
4502	SA	SA6	Alliance	Death Knight	7	1	36	44999	14648	675	1	\N	dps	1
4503	SA	SA6	Horde	Death Knight	3	4	22	89195	39462	237	\N	1	dps	1
4504	SA	SA6	Alliance	Shaman	5	1	71	111000	21765	732	1	\N	dps	1
4505	SA	SA6	Horde	Death Knight	0	10	18	46444	38058	223	\N	1	dps	1
4506	SA	SA6	Alliance	Mage	9	2	70	122000	14898	726	1	\N	dps	1
4507	SA	SA6	Alliance	Rogue	1	1	35	68510	4446	661	1	\N	dps	1
4508	SA	SA6	Horde	Druid	0	1	13	3161	27321	168	\N	1	heal	1
4509	SA	SA6	Alliance	Hunter	6	3	55	52896	7776	706	1	\N	dps	1
4510	SA	SA6	Horde	Shaman	0	7	20	30370	107000	235	\N	1	heal	1
4511	SA	SA6	Alliance	Priest	3	1	64	64523	135000	1049	1	\N	heal	1
4512	SA	SA6	Horde	Demon Hunter	3	9	25	69430	24870	242	\N	1	dps	1
4513	SA	SA6	Horde	Mage	4	4	23	72376	7290	239	\N	1	dps	1
4514	SA	SA6	Horde	Hunter	4	5	23	119000	7817	239	\N	1	dps	1
4515	SA	SA6	Alliance	Shaman	12	1	59	188000	7732	700	1	\N	dps	1
4516	SA	SA6	Alliance	Hunter	7	0	70	97494	8672	722	1	\N	dps	1
4517	SA	SA6	Horde	Priest	1	4	20	8570	8597	235	\N	1	heal	1
4518	SA	SA6	Alliance	Monk	4	3	71	26300	29905	726	1	\N	heal	1
4519	SA	SA6	Alliance	Monk	7	2	70	64303	22698	727	1	\N	dps	1
4520	SA	SA6	Horde	Priest	0	1	9	9378	46171	157	\N	1	heal	1
4521	SA	SA6	Horde	Death Knight	1	8	21	26038	9593	222	\N	1	dps	1
4522	SA	SA6	Horde	Warlock	1	4	20	36835	15934	231	\N	1	dps	1
4523	SA	SA6	Alliance	Paladin	1	0	67	4599	67918	711	1	\N	heal	1
4524	SA	SA6	Alliance	Druid	1	2	68	109000	9075	719	1	\N	dps	1
4525	SA	SA6	Horde	Shaman	6	5	24	105000	25400	241	\N	1	dps	1
4526	SA	SA6	Alliance	Warlock	9	1	55	77187	26581	707	1	\N	dps	1
4527	SA	SA6	Alliance	Shaman	6	2	65	87135	10680	1066	1	\N	dps	1
4528	SA	SA6	Horde	Priest	0	3	10	4914	97201	181	\N	1	heal	1
4529	SA	SA7	Alliance	Druid	0	5	28	12704	144000	317	\N	1	heal	\N
4530	SA	SA7	Horde	Druid	13	0	74	99853	45334	242	1	\N	dps	\N
4531	SA	SA7	Alliance	Paladin	0	5	37	16542	140000	222	\N	1	heal	\N
4532	SA	SA7	Horde	Paladin	5	5	66	70015	7905	229	1	\N	dps	\N
4533	SA	SA7	Horde	Paladin	10	1	57	153000	46310	522	1	\N	dps	\N
4534	SA	SA7	Alliance	Mage	8	8	37	111000	17085	333	\N	1	dps	\N
4535	SA	SA7	Horde	Monk	6	5	62	76278	36461	524	1	\N	dps	\N
4536	SA	SA7	Alliance	Druid	6	5	35	174000	49026	335	\N	1	dps	\N
4537	SA	SA7	Alliance	Mage	3	5	29	106000	16826	314	\N	1	dps	\N
4538	SA	SA7	Alliance	Warlock	3	3	31	91964	44403	316	\N	1	dps	\N
4539	SA	SA7	Alliance	Shaman	2	9	29	109000	9702	324	\N	1	dps	\N
4540	SA	SA7	Horde	Monk	2	5	66	97235	43108	381	1	\N	dps	\N
4541	SA	SA7	Alliance	Warrior	2	4	36	125000	27690	337	\N	1	dps	\N
4542	SA	SA7	Horde	Druid	0	2	65	18769	158000	231	1	\N	heal	\N
4543	SA	SA7	Alliance	Paladin	0	5	36	3121	210000	346	\N	1	heal	\N
4544	SA	SA7	Alliance	Paladin	7	1	19	121000	37003	165	\N	1	dps	\N
4545	SA	SA7	Horde	Shaman	2	5	58	30469	163000	517	1	\N	heal	\N
4546	SA	SA7	Horde	Paladin	1	4	73	18199	26255	543	1	\N	heal	\N
4547	SA	SA7	Alliance	Shaman	3	7	30	67038	14546	325	\N	1	dps	\N
4548	SA	SA7	Alliance	Hunter	1	5	30	51557	6610	325	\N	1	dps	\N
4549	SA	SA7	Horde	Mage	8	3	68	117000	4511	382	1	\N	dps	\N
4550	SA	SA7	Horde	Warlock	4	2	74	132000	48101	544	1	\N	dps	\N
4551	SA	SA7	Horde	Priest	0	2	72	43871	135000	543	1	\N	heal	\N
4552	SA	SA7	Horde	Hunter	6	2	72	111000	21021	540	1	\N	dps	\N
4553	SA	SA7	Horde	Warlock	7	5	48	82794	25987	519	1	\N	dps	\N
4554	SA	SA7	Alliance	Warrior	4	7	30	70396	16192	324	\N	1	dps	\N
4555	SA	SA7	Horde	Mage	7	2	66	153000	17751	379	1	\N	dps	\N
4556	SA	SA7	Alliance	Monk	2	5	30	54642	25826	330	\N	1	dps	\N
4557	SA	SA7	Alliance	Rogue	3	6	33	46141	5835	324	\N	1	dps	\N
4558	SA	SA7	Horde	Mage	5	2	50	79518	24327	523	1	\N	dps	\N
4559	BG	BG38	Horde	Warrior	0	1	0	11917	2548	135	\N	1	dps	\N
4560	BG	BG38	Alliance	Warlock	4	0	30	63901	14931	571	1	\N	dps	\N
4561	BG	BG38	Alliance	Paladin	0	0	30	2148	85150	788	1	\N	heal	\N
4562	BG	BG38	Horde	Warrior	0	3	0	17420	2489	115	\N	1	dps	\N
4563	BG	BG38	Horde	Paladin	0	5	1	51881	15150	138	\N	1	dps	\N
4564	BG	BG38	Horde	Shaman	0	7	1	8274	97489	137	\N	1	heal	\N
4565	BG	BG38	Alliance	Shaman	10	0	40	118000	0	595	1	\N	dps	\N
4566	BG	BG38	Alliance	Druid	0	2	25	1662	94881	550	1	\N	heal	\N
4567	BG	BG38	Horde	Death Knight	1	4	3	61292	27234	143	\N	1	dps	\N
4568	BG	BG38	Horde	Shaman	0	5	2	32555	10265	140	\N	1	dps	\N
4569	BG	BG38	Alliance	Warlock	1	0	37	57450	14866	584	1	\N	dps	\N
4570	BG	BG38	Alliance	Shaman	0	1	19	32664	1810	768	1	\N	dps	\N
4571	BG	BG38	Horde	Priest	0	5	1	14578	186000	138	\N	1	heal	\N
4572	BG	BG38	Horde	Death Knight	0	4	1	45225	11192	137	\N	1	dps	\N
4573	BG	BG38	Horde	Monk	0	3	3	34715	28980	143	\N	1	dps	\N
4574	BG	BG38	Horde	Death Knight	2	4	3	68694	14894	143	\N	1	dps	\N
4575	BG	BG38	Alliance	Warlock	5	0	38	111000	11676	808	1	\N	dps	\N
4576	BG	BG38	Alliance	Death Knight	6	0	40	105000	17535	589	1	\N	dps	\N
4577	BG	BG38	Alliance	Paladin	1	0	40	41671	107000	595	1	\N	heal	\N
4578	BG	BG38	Alliance	Warrior	12	0	25	55928	9661	773	1	\N	dps	\N
4579	BG	BG39	Alliance	Warrior	5	2	11	45820	20796	255	\N	1	dps	\N
4580	BG	BG39	Alliance	Mage	0	6	5	42267	14417	216	\N	1	dps	\N
4581	BG	BG39	Horde	Death Knight	2	0	33	44795	20042	391	1	\N	dps	\N
4582	BG	BG39	Alliance	Shaman	0	6	5	29484	14137	216	\N	1	dps	\N
4583	BG	BG39	Alliance	Rogue	0	3	8	34064	19865	229	\N	1	dps	\N
4584	BG	BG39	Alliance	Warlock	2	0	11	25909	15188	250	\N	1	dps	\N
4585	BG	BG39	Alliance	Druid	0	8	4	930	69154	213	\N	1	heal	\N
4586	BG	BG39	Alliance	Hunter	2	7	8	37303	10361	234	\N	1	dps	\N
4587	BG	BG39	Horde	Shaman	1	1	34	13160	90178	393	1	\N	heal	\N
4588	BG	BG39	Horde	Warlock	5	2	32	69133	20290	537	1	\N	dps	\N
4589	BG	BG39	Horde	Mage	1	0	11	23884	4124	344	1	\N	dps	\N
4590	BG	BG39	Horde	Rogue	7	3	33	58437	6232	540	1	\N	dps	\N
4591	BG	BG39	Horde	Monk	0	2	28	3651	31800	531	1	\N	heal	\N
4592	BG	BG39	Alliance	Priest	2	6	3	74685	27121	211	\N	1	dps	\N
4593	BG	BG39	Horde	Shaman	2	3	29	42640	16128	529	1	\N	dps	\N
4594	BG	BG39	Horde	Rogue	6	2	31	36387	11260	541	1	\N	dps	\N
4595	BG	BG39	Horde	Priest	10	1	37	84966	21644	546	1	\N	dps	\N
4596	BG	BG39	Horde	Priest	8	0	33	93391	16490	541	1	\N	dps	\N
4597	BG	BG39	Alliance	Druid	0	2	5	865	9833	217	\N	1	heal	\N
4598	BG	BG39	Alliance	Druid	3	3	11	31647	30464	248	\N	1	dps	\N
4599	BG	BG40	Alliance	Druid	0	4	13	14	21189	241	\N	1	heal	\N
4600	BG	BG40	Horde	Warrior	0	3	20	18969	1152	509	1	\N	dps	\N
4601	BG	BG40	Horde	Death Knight	3	2	27	25744	8212	527	1	\N	dps	\N
4602	BG	BG40	Alliance	Druid	1	4	9	16166	4473	227	\N	1	dps	\N
4603	BG	BG40	Alliance	Paladin	2	3	14	27735	13819	249	\N	1	dps	\N
4604	BG	BG40	Alliance	Demon Hunter	2	5	17	41613	11458	255	\N	1	dps	\N
4605	BG	BG40	Horde	Mage	7	3	24	32252	15981	374	1	\N	dps	\N
4606	BG	BG40	Horde	Warlock	8	2	32	60682	22269	545	1	\N	dps	\N
4607	BG	BG40	Alliance	Hunter	0	2	6	10793	1739	188	\N	1	dps	\N
4608	BG	BG40	Alliance	Hunter	1	5	14	32724	3497	243	\N	1	dps	\N
4609	BG	BG40	Horde	Shaman	3	3	38	13089	132000	397	1	\N	heal	\N
4610	BG	BG40	Horde	Rogue	7	2	32	33240	9253	533	1	\N	dps	\N
4611	BG	BG40	Horde	Druid	4	0	41	47544	8860	409	1	\N	dps	\N
4612	BG	BG40	Horde	Druid	6	1	36	40217	7360	541	1	\N	dps	\N
4613	BG	BG40	Alliance	Shaman	3	5	14	39894	11058	242	\N	1	dps	\N
4614	BG	BG40	Alliance	Warrior	2	3	6	25662	5805	222	\N	1	dps	\N
4615	BG	BG40	Horde	Warlock	6	1	38	60360	24603	553	1	\N	dps	\N
4616	BG	BG40	Alliance	Shaman	3	4	16	57858	5896	254	\N	1	dps	\N
4617	BG	BG40	Horde	Warlock	0	1	12	8952	8040	343	1	\N	dps	\N
4618	BG	BG40	Alliance	Demon Hunter	2	5	15	29027	11448	247	\N	1	dps	\N
4619	BG	BG41	Alliance	Hunter	0	5	2	33176	8658	192	\N	1	dps	\N
4620	BG	BG41	Alliance	Warrior	5	8	6	69126	23476	207	\N	1	dps	\N
4621	BG	BG41	Horde	Warrior	4	1	43	40354	5310	413	1	\N	dps	\N
4622	BG	BG41	Horde	Death Knight	8	0	37	84610	27439	549	1	\N	dps	\N
4623	BG	BG41	Horde	Shaman	0	0	11	14609	35854	347	1	\N	heal	\N
4624	BG	BG41	Alliance	Warrior	0	5	6	34986	4240	207	\N	1	dps	\N
4625	BG	BG41	Alliance	Priest	0	8	5	20058	19740	190	\N	1	dps	\N
4626	BG	BG41	Horde	Rogue	4	4	31	58767	12252	391	1	\N	dps	\N
4627	BG	BG41	Alliance	Priest	0	7	2	12077	71972	192	\N	1	heal	\N
4628	BG	BG41	Alliance	Rogue	0	1	6	13899	3671	207	\N	1	dps	\N
4629	BG	BG41	Alliance	Warlock	0	7	4	71939	23184	199	\N	1	dps	\N
4630	BG	BG41	Horde	Priest	1	0	43	32943	132000	412	1	\N	heal	\N
4631	BG	BG41	Horde	Warrior	6	0	19	32993	13633	511	1	\N	dps	\N
4632	BG	BG41	Alliance	Druid	0	6	2	35047	14185	192	\N	1	dps	\N
4633	BG	BG41	Horde	Paladin	6	0	29	51702	10609	525	1	\N	dps	\N
4634	BG	BG41	Horde	Mage	11	0	45	73707	9985	415	1	\N	dps	\N
4635	BG	BG41	Alliance	Druid	1	2	5	6484	9176	204	\N	1	heal	\N
4636	BG	BG41	Alliance	Warrior	0	5	3	12490	5660	197	\N	1	dps	\N
4637	BG	BG41	Horde	Warrior	7	1	43	46532	10505	419	1	\N	dps	\N
4638	BG	BG41	Horde	Paladin	0	0	26	48640	14522	530	1	\N	dps	\N
4639	BG	BG42	Alliance	Shaman	1	0	5	14303	5399	834	1	\N	dps	1
4640	BG	BG42	Horde	Shaman	3	4	4	30847	16759	199	\N	1	dps	1
4641	BG	BG42	Alliance	Death Knight	12	1	56	149000	26698	1084	1	\N	dps	1
4642	BG	BG42	Alliance	Warlock	18	0	57	152000	22352	1089	1	\N	dps	1
4643	BG	BG42	Horde	Warrior	2	12	7	39212	5456	200	\N	1	dps	1
4644	BG	BG42	Horde	Shaman	0	11	1	15465	101000	186	\N	1	heal	1
4645	BG	BG42	Horde	Demon Hunter	1	4	3	31769	17941	196	\N	1	dps	1
4646	BG	BG42	Alliance	Hunter	8	0	49	90389	3920	737	1	\N	dps	1
4647	BG	BG42	Horde	Mage	1	5	5	61647	7549	194	\N	1	dps	1
4648	BG	BG42	Alliance	Mage	4	1	55	68222	6983	742	1	\N	dps	1
4649	BG	BG42	Alliance	Shaman	0	0	56	8293	172000	754	1	\N	heal	1
4650	BG	BG42	Horde	Warrior	2	9	5	28573	6030	194	\N	1	dps	1
4651	BG	BG42	Horde	Priest	0	2	0	35529	80757	152	\N	1	heal	1
4652	BG	BG42	Alliance	Rogue	3	1	31	55062	8212	1033	1	\N	dps	1
4653	BG	BG42	Alliance	Rogue	5	3	43	33594	9957	700	1	\N	dps	1
4654	BG	BG42	Alliance	Mage	5	0	55	47337	1497	1079	1	\N	dps	1
4655	BG	BG42	Horde	Mage	0	5	5	59232	29288	194	\N	1	dps	1
4656	BG	BG42	Horde	Warrior	0	4	2	8688	1536	189	\N	1	dps	1
4657	BG	BG42	Alliance	Paladin	5	3	42	47939	21364	1037	1	\N	dps	1
4658	BG	BG42	Horde	Druid	0	4	7	262	63689	200	\N	1	heal	1
4659	DG	DG7	Alliance	Paladin	0	6	14	1301	32519	360	\N	1	heal	\N
4660	DG	DG7	Alliance	Druid	2	2	25	92459	23224	389	\N	1	dps	\N
4661	DG	DG7	Horde	Priest	4	2	45	33649	108000	436	1	\N	heal	\N
4662	DG	DG7	Horde	Hunter	2	1	48	46983	5533	437	1	\N	dps	\N
4663	DG	DG7	Alliance	Death Knight	4	2	25	89854	45154	389	\N	1	dps	\N
4664	DG	DG7	Horde	Warrior	9	2	47	115000	25790	426	1	\N	dps	\N
4665	DG	DG7	Alliance	Demon Hunter	1	8	18	69992	14492	366	\N	1	dps	\N
4666	DG	DG7	Horde	Hunter	5	4	34	42936	10297	415	1	\N	dps	\N
4667	DG	DG7	Horde	Paladin	7	3	34	76520	25227	409	1	\N	dps	\N
4668	DG	DG7	Alliance	Paladin	3	3	19	79551	30500	373	\N	1	dps	\N
4669	DG	DG7	Alliance	Rogue	2	1	15	40493	11818	372	\N	1	dps	\N
4670	DG	DG7	Horde	Monk	1	2	46	3659	103000	575	1	\N	heal	\N
4671	DG	DG7	Alliance	Warlock	2	3	24	49550	21463	393	\N	1	dps	\N
4672	DG	DG7	Horde	Shaman	0	2	17	6264	26442	329	1	\N	heal	\N
4673	DG	DG7	Horde	Rogue	4	5	29	51832	5118	544	1	\N	dps	\N
4674	DG	DG7	Horde	Priest	3	0	36	42997	19836	382	1	\N	dps	\N
4675	DG	DG7	Horde	Mage	3	3	42	44988	6049	568	1	\N	dps	\N
4676	DG	DG7	Alliance	Paladin	2	6	16	43382	4365	368	\N	1	dps	\N
4677	DG	DG7	Alliance	Monk	0	4	11	1649	50125	352	\N	1	heal	\N
4678	DG	DG7	Horde	Death Knight	2	1	40	28758	32037	574	1	\N	dps	\N
4679	DG	DG7	Alliance	Paladin	2	7	23	52263	16942	388	\N	1	dps	\N
4680	DG	DG7	Alliance	Hunter	1	5	20	34691	10242	377	\N	1	dps	\N
4681	DG	DG7	Alliance	Warlock	2	2	26	165000	50531	397	\N	1	dps	\N
4682	DG	DG7	Horde	Warlock	6	3	41	56943	35698	572	1	\N	dps	\N
4683	DG	DG7	Alliance	Warrior	7	3	20	60589	15457	381	\N	1	dps	\N
4684	DG	DG7	Horde	Priest	0	0	40	5329	69404	418	1	\N	heal	\N
4685	DG	DG7	Horde	Monk	2	0	46	4581	159000	576	1	\N	heal	\N
4686	DG	DG7	Horde	Shaman	9	3	33	91020	23466	418	1	\N	dps	\N
4687	DG	DG7	Alliance	Priest	0	4	18	27453	45385	372	\N	1	heal	\N
4688	DG	DG7	Alliance	Rogue	5	1	20	47645	6252	372	\N	1	dps	\N
4689	DG	DG8	Horde	Warlock	1	5	14	57443	28676	170	\N	1	dps	1
4690	DG	DG8	Horde	Mage	3	4	10	12191	14884	61	\N	1	dps	1
4691	DG	DG8	Alliance	Mage	4	3	37	69611	9949	695	1	\N	dps	1
4692	DG	DG8	Alliance	Shaman	4	1	48	42778	4660	728	1	\N	dps	1
4693	DG	DG8	Alliance	Priest	0	0	52	24124	153000	1068	1	\N	heal	1
4694	DG	DG8	Alliance	Paladin	5	0	49	150000	30278	1068	1	\N	dps	1
4695	DG	DG8	Horde	Warrior	0	1	15	18900	2895	174	\N	1	dps	1
4696	DG	DG8	Horde	Paladin	3	3	9	43070	19440	164	\N	1	dps	1
4697	DG	DG8	Horde	Paladin	0	3	20	22317	140000	178	\N	1	heal	1
4698	DG	DG8	Alliance	Priest	1	2	49	4763	198000	1064	1	\N	heal	1
4699	DG	DG8	Horde	Shaman	1	5	16	25056	137000	171	\N	1	heal	1
4700	DG	DG8	Horde	Priest	0	5	18	18668	72885	67	\N	1	heal	1
4701	DG	DG8	Alliance	Warrior	9	1	47	54519	17024	1057	1	\N	dps	1
4702	DG	DG8	Horde	Warrior	3	6	16	39579	14599	174	\N	1	dps	1
4703	DG	DG8	Alliance	Death Knight	5	0	51	39397	5143	743	1	\N	dps	1
4704	DG	DG8	Alliance	Mage	2	2	50	76564	4179	1072	1	\N	dps	1
4705	DG	DG8	Horde	Mage	1	3	15	29884	5894	172	\N	1	dps	1
4706	DG	DG8	Alliance	Mage	2	2	38	64478	13138	1034	1	\N	dps	1
4707	DG	DG8	Horde	Demon Hunter	3	3	17	76649	21046	187	\N	1	dps	1
4708	DG	DG8	Alliance	Mage	7	1	52	89109	6004	1073	1	\N	dps	1
4709	DG	DG8	Alliance	Druid	2	3	31	66718	4707	681	1	\N	dps	1
4710	DG	DG8	Horde	Death Knight	2	4	18	75755	17614	174	\N	1	dps	1
4711	DG	DG8	Alliance	Warrior	5	4	49	78569	16698	721	1	\N	dps	1
4712	DG	DG8	Alliance	Warrior	5	3	34	29727	3713	1029	1	\N	dps	1
4713	DG	DG8	Alliance	Druid	0	1	39	955	60178	367	1	\N	heal	1
4714	DG	DG8	Alliance	Hunter	5	1	47	93622	5811	724	1	\N	dps	1
4715	DG	DG8	Horde	Druid	2	3	20	58851	19152	178	\N	1	dps	1
4716	DG	DG8	Horde	Shaman	3	4	19	60023	2795	177	\N	1	dps	1
4717	DG	DG8	Horde	Demon Hunter	0	3	21	66294	18595	180	\N	1	dps	1
4718	DG	DG8	Horde	Death Knight	0	6	7	47352	17459	164	\N	1	dps	1
4719	ES	ES18	Alliance	Hunter	4	4	46	92363	11015	680	1	\N	dps	\N
4720	ES	ES18	Alliance	Warrior	11	6	45	87688	26122	910	1	\N	dps	\N
4721	ES	ES18	Horde	Death Knight	7	3	38	55830	22352	321	\N	1	dps	\N
4722	ES	ES18	Horde	Druid	2	2	11	33564	8594	204	\N	1	dps	\N
4723	ES	ES18	Alliance	Death Knight	9	1	42	91504	23342	927	1	\N	dps	\N
4724	ES	ES18	Horde	Druid	1	4	35	57012	37525	332	\N	1	dps	\N
4725	ES	ES18	Alliance	Priest	0	4	40	2846	135000	911	1	\N	heal	\N
4726	ES	ES18	Alliance	Rogue	6	4	45	36044	9470	897	1	\N	dps	\N
4727	ES	ES18	Alliance	Death Knight	3	4	35	67142	46404	899	1	\N	dps	\N
4728	ES	ES18	Alliance	Druid	0	5	51	25910	35991	926	1	\N	heal	\N
4729	ES	ES18	Horde	Warlock	3	2	46	96845	16413	330	\N	1	dps	\N
4730	ES	ES18	Horde	Shaman	1	5	28	25122	124000	312	\N	1	heal	\N
4731	ES	ES18	Horde	Warrior	2	2	13	27231	5707	202	\N	1	dps	\N
4732	ES	ES18	Alliance	Demon Hunter	5	2	45	66497	20950	927	1	\N	dps	\N
4733	ES	ES18	Alliance	Priest	0	3	46	11801	139000	923	1	\N	heal	\N
4734	ES	ES18	Horde	Priest	0	6	50	18543	84249	345	\N	1	heal	\N
4735	ES	ES18	Alliance	Warlock	3	3	44	40570	31402	933	1	\N	dps	\N
4736	ES	ES18	Alliance	Death Knight	7	7	50	125000	32378	925	1	\N	dps	\N
4737	ES	ES18	Horde	Rogue	1	5	41	33844	3404	328	\N	1	dps	\N
4738	ES	ES18	Alliance	Hunter	2	3	40	40710	9585	924	1	\N	dps	\N
4739	ES	ES18	Horde	Mage	4	2	15	21259	8482	214	\N	1	dps	\N
4740	ES	ES18	Horde	Mage	5	6	32	71283	28731	309	\N	1	dps	\N
4741	ES	ES18	Horde	Priest	4	4	40	78060	26557	327	\N	1	dps	\N
4742	ES	ES18	Horde	Warrior	11	4	43	115000	37006	336	\N	1	dps	\N
4743	ES	ES18	Horde	Priest	1	5	31	16039	77040	315	\N	1	heal	\N
4744	ES	ES18	Horde	Rogue	4	6	38	58690	11181	332	\N	1	dps	\N
4745	ES	ES18	Alliance	Monk	4	7	39	71170	8805	903	1	\N	dps	\N
4746	ES	ES18	Alliance	Warrior	7	3	42	66795	17809	909	1	\N	dps	\N
4747	ES	ES18	Alliance	Warlock	5	5	38	86945	33977	906	1	\N	dps	\N
4748	ES	ES18	Horde	Death Knight	4	4	35	58609	30663	321	\N	1	dps	\N
4749	SS	SS6	Horde	Paladin	3	2	19	47185	18195	245	\N	1	dps	\N
4750	SS	SS6	Horde	Death Knight	3	2	18	50542	74320	241	\N	1	dps	\N
4751	SS	SS6	Alliance	Priest	1	3	16	16626	15873	752	1	\N	dps	\N
4752	SS	SS6	Horde	Monk	1	2	19	1269	81627	242	\N	1	heal	\N
4753	SS	SS6	Alliance	Priest	0	1	24	6701	83113	774	1	\N	heal	\N
4754	SS	SS6	Alliance	Death Knight	2	1	23	59800	11925	996	1	\N	dps	\N
4755	SS	SS6	Horde	Shaman	0	2	7	10604	25325	159	\N	1	heal	\N
4756	SS	SS6	Alliance	Warlock	5	3	21	56444	24127	543	1	\N	dps	\N
4757	SS	SS6	Horde	Rogue	3	2	20	58668	13537	245	\N	1	dps	\N
4758	SS	SS6	Alliance	Paladin	4	4	20	58033	20046	762	1	\N	dps	\N
4759	SS	SS6	Alliance	Death Knight	2	3	20	55706	17048	761	1	\N	dps	\N
4760	SS	SS6	Horde	Paladin	2	1	21	13828	14789	248	\N	1	heal	\N
4761	SS	SS6	Alliance	Warrior	4	1	22	27808	9720	541	1	\N	dps	\N
4762	SS	SS6	Horde	Monk	1	3	16	27662	9254	238	\N	1	dps	\N
4763	SS	SS6	Horde	Demon Hunter	1	4	12	33853	6953	228	\N	1	dps	\N
4764	SS	SS6	Alliance	Shaman	2	3	14	44074	22997	516	1	\N	dps	\N
4765	SS	SS6	Alliance	Paladin	0	1	17	52165	9684	529	1	\N	dps	\N
4766	SS	SS6	Alliance	Priest	3	3	22	40583	4929	768	1	\N	dps	\N
4767	SS	SS6	Horde	Shaman	5	4	16	47112	21966	235	\N	1	dps	\N
4768	SS	SS6	Horde	Death Knight	3	1	21	39123	17104	248	\N	1	dps	\N
4769	SS	SS7	Horde	Hunter	1	3	9	96410	13657	247	\N	1	dps	\N
4770	SS	SS7	Alliance	Priest	7	0	40	120000	33861	1020	1	\N	dps	\N
4771	SS	SS7	Alliance	Priest	9	0	43	88729	31080	1035	1	\N	dps	\N
4772	SS	SS7	Alliance	Priest	1	1	36	6531	93102	1010	1	\N	heal	\N
4773	SS	SS7	Alliance	Druid	1	3	31	30194	28582	770	1	\N	dps	\N
4774	SS	SS7	Alliance	Warrior	7	1	38	50707	11206	1014	1	\N	dps	\N
4775	SS	SS7	Horde	Shaman	1	5	11	13655	77257	253	\N	1	heal	\N
4776	SS	SS7	Alliance	Warlock	2	4	30	58107	48831	769	1	\N	dps	\N
4777	SS	SS7	Horde	Priest	0	1	10	2588	80074	250	\N	1	heal	\N
4778	SS	SS7	Horde	Priest	3	3	13	103000	20079	252	\N	1	dps	\N
4779	SS	SS7	Horde	Shaman	0	2	4	18380	2779	137	\N	1	dps	\N
4780	SS	SS7	Alliance	Druid	5	1	42	77114	20293	1025	1	\N	dps	\N
4781	SS	SS7	Horde	Demon Hunter	2	5	9	66520	12583	248	\N	1	dps	\N
4782	SS	SS7	Horde	Warlock	2	5	6	70324	27286	240	\N	1	dps	\N
4783	SS	SS7	Alliance	Hunter	1	2	35	39316	3834	1013	1	\N	dps	\N
4784	SS	SS7	Alliance	Warrior	9	2	42	58599	4890	808	1	\N	dps	\N
4785	SS	SS7	Horde	Priest	0	8	8	40713	24839	246	\N	1	dps	\N
4786	SS	SS7	Horde	Druid	0	5	5	28258	35560	238	\N	1	heal	\N
4787	SS	SS7	Horde	Druid	5	4	9	80759	14141	247	\N	1	dps	\N
4788	SS	SS7	Alliance	Monk	1	0	39	2565	170000	1017	1	\N	heal	\N
4789	SS	SS8	Horde	Paladin	1	5	13	8375	99774	371	\N	1	heal	\N
4790	SS	SS8	Horde	Mage	4	2	24	78785	16130	395	\N	1	dps	\N
4791	SS	SS8	Alliance	Rogue	3	4	32	70267	10144	803	1	\N	dps	\N
4792	SS	SS8	Alliance	Death Knight	8	4	30	70889	26487	1022	1	\N	dps	\N
4793	SS	SS8	Horde	Paladin	0	6	19	52220	21897	383	\N	1	dps	\N
4794	SS	SS8	Horde	Druid	2	3	21	74775	29189	388	\N	1	dps	\N
4795	SS	SS8	Alliance	Druid	1	1	34	13954	208000	1039	1	\N	heal	\N
4796	SS	SS8	Alliance	Mage	6	4	27	69413	22892	793	1	\N	dps	\N
4797	SS	SS8	Alliance	Priest	2	1	26	24010	91626	787	1	\N	heal	\N
4798	SS	SS8	Horde	Paladin	4	4	25	81299	25477	397	\N	1	dps	\N
4799	SS	SS8	Horde	Demon Hunter	4	2	21	75936	15317	388	\N	1	dps	\N
4800	SS	SS8	Horde	Shaman	0	7	21	25427	105000	389	\N	1	heal	\N
4801	SS	SS8	Horde	Hunter	3	4	25	51027	9462	397	\N	1	dps	\N
4802	SS	SS8	Horde	Priest	2	6	13	115000	19603	372	\N	1	dps	\N
4803	SS	SS8	Alliance	Hunter	4	3	20	69465	8561	990	1	\N	dps	\N
4804	SS	SS8	Alliance	Mage	3	4	27	47178	9378	1014	1	\N	dps	\N
4805	SS	SS8	Alliance	Mage	3	4	22	30211	3222	994	1	\N	dps	\N
4806	SS	SS8	Alliance	Death Knight	8	2	32	105000	33720	803	1	\N	dps	\N
4807	SS	SS8	Horde	Hunter	4	3	25	38208	8013	397	\N	1	dps	\N
4808	SS	SS8	Alliance	Druid	3	2	31	59346	14084	803	1	\N	dps	\N
4809	TK	TK33	Alliance	Paladin	0	4	7	7611	5743	233	\N	1	dps	\N
4810	TK	TK33	Alliance	Warrior	0	7	5	50091	1398	227	\N	1	dps	\N
4811	TK	TK33	Horde	Monk	0	0	34	1554	127000	518	1	\N	heal	\N
4812	TK	TK33	Horde	Priest	1	0	34	36620	119000	518	1	\N	heal	\N
4813	TK	TK33	Horde	Shaman	3	0	34	131000	3362	368	1	\N	dps	\N
4814	TK	TK33	Horde	Shaman	1	1	31	15474	74199	512	1	\N	heal	\N
4815	TK	TK33	Alliance	Paladin	2	3	6	38758	8862	230	\N	1	dps	\N
4816	TK	TK33	Alliance	Druid	0	1	7	0	97298	230	\N	1	heal	\N
4817	TK	TK33	Alliance	Paladin	0	4	6	31703	23663	230	\N	1	dps	\N
4818	TK	TK33	Alliance	Priest	0	0	6	37361	71316	231	\N	1	heal	\N
4819	TK	TK33	Alliance	Priest	2	3	5	157000	21982	229	\N	1	dps	\N
4820	TK	TK33	Horde	Priest	15	1	32	103000	15377	514	1	\N	dps	\N
4821	TK	TK33	Horde	Mage	2	0	34	89327	6970	518	1	\N	dps	\N
4822	TK	TK33	Alliance	Monk	1	4	7	28943	25171	233	\N	1	dps	\N
4823	TK	TK33	Horde	Rogue	1	2	33	35745	9200	516	1	\N	dps	\N
4824	TK	TK33	Alliance	Mage	1	6	6	44724	12841	230	\N	1	dps	\N
4825	TK	TK33	Horde	Warrior	5	1	30	40888	3933	510	1	\N	dps	\N
4826	TK	TK33	Horde	Druid	2	2	30	27564	8466	510	1	\N	dps	\N
4827	TK	TK33	Alliance	Shaman	1	3	3	5175	54224	193	\N	1	heal	\N
4828	TK	TK33	Horde	Warrior	2	0	34	32325	5893	518	1	\N	dps	\N
4829	TK	TK34	Alliance	Druid	0	4	20	37452	34708	255	\N	1	dps	\N
4830	TK	TK34	Horde	Hunter	4	0	43	153000	6715	541	1	\N	dps	\N
4831	TK	TK34	Alliance	Rogue	2	1	26	27207	2647	379	\N	1	dps	\N
4832	TK	TK34	Alliance	Monk	0	2	26	4657	209000	379	\N	1	heal	\N
4833	TK	TK34	Alliance	Rogue	3	5	23	100000	12499	372	\N	1	dps	\N
4834	TK	TK34	Horde	Shaman	1	3	41	29498	160000	387	1	\N	heal	\N
4835	TK	TK34	Horde	Mage	6	1	43	111000	11156	541	1	\N	dps	\N
4836	TK	TK34	Alliance	Warlock	8	4	26	90267	39885	379	\N	1	dps	\N
4837	TK	TK34	Alliance	Druid	6	5	22	110000	14997	367	\N	1	dps	\N
4838	TK	TK34	Alliance	Warrior	0	5	24	26646	42153	373	\N	1	dps	\N
4839	TK	TK34	Horde	Priest	1	5	38	37001	99769	381	1	\N	heal	\N
4840	TK	TK34	Horde	Warrior	4	1	43	113000	38077	391	1	\N	dps	\N
4841	TK	TK34	Alliance	Warrior	4	7	20	50723	1533	364	\N	1	dps	\N
4842	TK	TK34	Horde	Death Knight	3	3	35	104000	24877	525	1	\N	dps	\N
4843	TK	TK34	Horde	Warlock	2	6	34	48301	30790	523	1	\N	dps	\N
4844	TK	TK34	Horde	Mage	4	0	35	69845	8525	487	1	\N	dps	\N
4845	TK	TK34	Alliance	Druid	0	3	25	898	242000	377	\N	1	heal	\N
4846	TK	TK34	Horde	Mage	13	4	36	99191	10380	527	1	\N	dps	\N
4847	TK	TK34	Alliance	Warrior	3	7	19	105000	20603	362	\N	1	dps	\N
4848	TK	TK34	Horde	Paladin	4	2	39	128000	30625	533	1	\N	dps	\N
4849	TK	TK35	Alliance	Warlock	10	2	48	84576	29988	586	1	\N	dps	\N
4850	TK	TK35	Alliance	Priest	8	0	50	109000	19839	590	1	\N	dps	\N
4851	TK	TK35	Alliance	Druid	0	1	50	3738	143000	590	1	\N	heal	\N
4852	TK	TK35	Alliance	Warrior	7	3	47	40081	6069	583	1	\N	dps	\N
4853	TK	TK35	Horde	Rogue	0	7	15	30002	10648	200	\N	1	dps	\N
4854	TK	TK35	Horde	Shaman	0	5	19	18937	100000	208	\N	1	heal	\N
4855	TK	TK35	Alliance	Monk	1	3	45	2573	85092	575	1	\N	heal	\N
4856	TK	TK35	Alliance	Hunter	3	2	46	55281	2831	581	1	\N	dps	\N
4857	TK	TK35	Alliance	Warrior	1	5	35	42471	5694	782	1	\N	dps	\N
4858	TK	TK35	Horde	Mage	4	1	22	61820	25728	214	\N	1	dps	\N
4859	TK	TK35	Horde	Shaman	2	8	16	32711	9954	202	\N	1	dps	\N
4860	TK	TK35	Horde	Shaman	1	4	20	37479	6396	210	\N	1	dps	\N
4861	TK	TK35	Horde	Warrior	3	5	20	54201	9354	210	\N	1	dps	\N
4862	TK	TK35	Horde	Monk	1	4	18	3490	43846	206	\N	1	heal	\N
4863	TK	TK35	Alliance	Hunter	6	2	41	41153	13044	564	1	\N	dps	\N
4864	TK	TK35	Horde	Mage	3	5	17	80002	23565	204	\N	1	dps	\N
4865	TK	TK35	Alliance	Druid	8	0	50	102000	23841	590	1	\N	dps	\N
4866	TK	TK35	Horde	Death Knight	6	5	20	119000	23228	210	\N	1	dps	\N
4867	TK	TK35	Alliance	Druid	4	4	46	45510	10073	804	1	\N	dps	\N
4868	TK	TK35	Horde	Shaman	1	6	18	41611	16393	206	\N	1	dps	\N
4869	TK	TK36	Alliance	Shaman	6	4	24	52695	25345	292	\N	1	dps	\N
4870	TK	TK36	Horde	Druid	0	4	27	28108	20113	362	1	\N	dps	\N
4871	TK	TK36	Horde	Hunter	4	2	32	71977	1514	522	1	\N	dps	\N
4872	TK	TK36	Alliance	Druid	0	4	20	27415	12355	279	\N	1	dps	\N
4873	TK	TK36	Horde	Mage	3	3	33	50517	13317	374	1	\N	dps	\N
4874	TK	TK36	Horde	Shaman	0	0	34	17636	109000	526	1	\N	heal	\N
4875	TK	TK36	Alliance	Rogue	3	3	23	50239	18997	286	\N	1	dps	\N
4876	TK	TK36	Horde	Shaman	0	4	30	19629	87569	518	1	\N	heal	\N
4877	TK	TK36	Horde	Warrior	9	2	32	91965	20738	372	1	\N	dps	\N
4878	TK	TK36	Horde	Demon Hunter	1	3	31	21503	2888	520	1	\N	dps	\N
4879	TK	TK36	Alliance	Demon Hunter	4	2	25	85229	17286	294	\N	1	dps	\N
4880	TK	TK36	Horde	Priest	10	1	33	212000	21176	524	1	\N	dps	\N
4881	TK	TK36	Horde	Warrior	5	5	30	65103	10781	368	1	\N	dps	\N
4882	TK	TK36	Alliance	Paladin	7	6	22	77932	28740	286	\N	1	dps	\N
4883	TK	TK36	Alliance	Paladin	0	0	27	1281	189000	296	\N	1	heal	\N
4884	TK	TK36	Alliance	Paladin	4	3	24	121000	31378	290	\N	1	dps	\N
4885	TK	TK36	Alliance	Mage	1	2	21	25849	8216	267	\N	1	dps	\N
4886	TK	TK36	Alliance	Mage	2	5	18	53364	7965	275	\N	1	dps	\N
4887	TK	TK36	Horde	Priest	2	3	26	11844	72376	510	1	\N	heal	\N
4888	TK	TK36	Alliance	Monk	0	5	21	6664	63217	280	\N	1	heal	\N
4889	TP	TP24	Horde	Shaman	7	3	24	119000	32782	196	\N	1	dps	\N
4890	TP	TP24	Horde	Mage	1	4	18	100000	22233	185	\N	1	dps	\N
4891	TP	TP24	Alliance	Warlock	9	0	55	114000	24252	636	1	\N	dps	\N
4892	TP	TP24	Alliance	Warlock	8	4	51	95000	32956	849	1	\N	dps	\N
4893	TP	TP24	Horde	Shaman	0	8	15	18416	157000	176	\N	1	heal	\N
4894	TP	TP24	Alliance	Warrior	15	1	51	152000	37830	844	1	\N	dps	\N
4895	TP	TP24	Alliance	Rogue	8	2	43	76820	13389	825	1	\N	dps	\N
4896	TP	TP24	Horde	Warrior	6	10	15	46259	7370	179	\N	1	dps	\N
4897	TP	TP24	Alliance	Death Knight	7	4	46	128000	44140	831	1	\N	dps	\N
4898	TP	TP24	Alliance	Warrior	6	4	41	59827	14672	809	1	\N	dps	\N
4899	TP	TP24	Alliance	Mage	7	4	51	124000	6801	840	1	\N	dps	\N
4900	TP	TP24	Alliance	Paladin	0	2	49	2617	167000	843	1	\N	heal	\N
4901	TP	TP24	Horde	Warlock	1	8	15	74555	67191	175	\N	1	dps	\N
4902	TP	TP24	Horde	Warlock	0	3	7	23858	12858	119	\N	1	dps	\N
4903	TP	TP24	Alliance	Druid	1	1	52	4858	281000	845	1	\N	heal	\N
4904	TP	TP24	Alliance	Demon Hunter	3	4	41	56661	9688	583	1	\N	dps	\N
4905	TP	TP24	Horde	Mage	3	5	19	90463	17017	185	\N	1	dps	\N
4906	TP	TP24	Horde	Druid	1	6	21	82700	11312	190	\N	1	dps	\N
4907	TP	TP24	Horde	Paladin	5	8	22	96655	28330	192	\N	1	dps	\N
4908	TP	TP24	Horde	Druid	1	6	20	56592	43499	187	\N	1	dps	\N
4909	TP	TP25	Alliance	Hunter	4	2	11	43848	16223	271	\N	1	dps	1
4910	TP	TP25	Horde	Shaman	10	2	28	51918	18143	462	1	\N	dps	1
4911	TP	TP25	Alliance	Druid	2	5	7	31745	7522	252	\N	1	dps	1
4912	TP	TP25	Horde	Shaman	5	1	31	88732	12113	479	1	\N	dps	1
4913	TP	TP25	Alliance	Paladin	1	6	8	20053	4357	258	\N	1	dps	1
4914	TP	TP25	Horde	Druid	1	2	16	22429	9349	447	1	\N	dps	1
4915	TP	TP25	Alliance	Death Knight	1	3	10	47174	17004	274	\N	1	dps	1
4916	TP	TP25	Horde	Warrior	5	3	23	28098	3715	449	1	\N	dps	1
4917	TP	TP25	Horde	Shaman	0	2	15	7097	42227	445	1	\N	heal	1
4918	TP	TP25	Alliance	Rogue	1	1	11	17031	2076	271	\N	1	dps	1
4919	TP	TP25	Horde	Rogue	2	1	27	27813	5679	454	1	\N	dps	1
4920	TP	TP25	Horde	Rogue	1	1	28	33688	9619	684	1	\N	dps	1
4921	TP	TP25	Alliance	Druid	0	4	10	12997	19038	269	\N	1	heal	1
4922	TP	TP25	Alliance	Hunter	2	3	9	40645	13736	264	\N	1	dps	1
4923	TP	TP25	Horde	Warrior	5	3	25	32651	10611	453	1	\N	dps	1
4924	TP	TP25	Alliance	Rogue	4	3	8	40786	12341	259	\N	1	dps	1
4925	TP	TP25	Horde	Shaman	4	1	19	35607	18551	664	1	\N	dps	1
4926	TP	TP25	Horde	Shaman	0	2	17	1767	71046	435	1	\N	heal	1
4927	TP	TP25	Alliance	Warlock	3	1	12	71576	18054	294	\N	1	dps	1
4928	TP	TP26	Alliance	Paladin	0	1	31	623	69320	306	\N	1	heal	\N
4929	TP	TP26	Horde	Mage	2	5	23	61149	17803	329	1	\N	dps	\N
4930	TP	TP26	Alliance	Paladin	5	7	29	85434	25430	306	\N	1	dps	\N
4931	TP	TP26	Horde	Death Knight	9	4	23	69296	39877	327	1	\N	dps	\N
4932	TP	TP26	Horde	Paladin	2	6	22	104000	42276	477	1	\N	dps	\N
4933	TP	TP26	Alliance	Mage	5	2	21	66927	7766	279	\N	1	dps	\N
4934	TP	TP26	Alliance	Hunter	10	0	39	113000	9861	350	\N	1	dps	\N
4935	TP	TP26	Horde	Druid	0	4	20	32682	32452	469	1	\N	dps	\N
4936	TP	TP26	Alliance	Druid	6	5	28	42210	4459	296	\N	1	dps	\N
4937	TP	TP26	Horde	Shaman	1	4	22	20209	128000	474	1	\N	heal	\N
4938	TP	TP26	Alliance	Priest	0	3	26	12395	97101	287	\N	1	heal	\N
4939	TP	TP26	Horde	Priest	0	2	15	3067	4617	311	1	\N	heal	\N
4940	TP	TP26	Horde	Paladin	5	6	21	61822	30752	473	1	\N	dps	\N
4941	TP	TP26	Alliance	Paladin	6	2	40	89804	35855	352	\N	1	dps	\N
4942	TP	TP26	Horde	Rogue	0	6	9	17178	12626	297	1	\N	dps	\N
4943	TP	TP26	Alliance	Demon Hunter	4	7	25	77045	16977	290	\N	1	dps	\N
4944	TP	TP26	Horde	Shaman	7	5	22	58562	27483	477	1	\N	dps	\N
4945	TP	TP26	Horde	Paladin	3	4	17	34603	23084	463	1	\N	dps	\N
4946	TP	TP26	Alliance	Druid	1	1	29	40800	10906	299	\N	1	dps	\N
4947	TP	TP26	Alliance	Druid	8	1	42	80905	26023	355	\N	1	dps	\N
4948	WG	WG40	Alliance	Druid	0	4	28	49626	16990	285	\N	1	dps	\N
4949	WG	WG40	Alliance	Shaman	0	5	34	12090	98080	307	\N	1	heal	\N
4950	WG	WG40	Horde	Druid	4	0	12	48444	17459	277	1	\N	dps	\N
4951	WG	WG40	Horde	Death Knight	6	6	32	117000	42210	386	1	\N	dps	\N
4952	WG	WG40	Alliance	Druid	0	5	26	1495	132000	279	\N	1	heal	\N
4953	WG	WG40	Horde	Shaman	1	3	31	10437	107000	380	1	\N	heal	\N
4954	WG	WG40	Alliance	Mage	3	5	32	100000	18330	301	\N	1	dps	\N
4955	WG	WG40	Alliance	Mage	9	6	32	125000	13100	301	\N	1	dps	\N
4956	WG	WG40	Alliance	Shaman	3	5	32	118000	11833	300	\N	1	dps	\N
4957	WG	WG40	Horde	Death Knight	2	5	31	48443	63386	378	1	\N	dps	\N
4958	WG	WG40	Alliance	Rogue	5	1	33	40052	9265	303	\N	1	dps	\N
4959	WG	WG40	Horde	Warlock	12	3	38	98418	37005	249	1	\N	dps	\N
4960	WG	WG40	Horde	Warlock	3	7	35	57621	30235	537	1	\N	dps	\N
4961	WG	WG40	Horde	Warlock	4	7	29	79408	69097	526	1	\N	dps	\N
4962	WG	WG40	Alliance	Mage	1	4	27	93972	26411	280	\N	1	dps	\N
4963	WG	WG40	Horde	Monk	8	1	37	90969	36229	395	1	\N	dps	\N
4964	WG	WG40	Horde	Monk	1	2	32	10892	113000	530	1	\N	heal	\N
4965	WG	WG40	Alliance	Death Knight	10	5	32	150000	42645	302	\N	1	dps	\N
4966	WG	WG40	Alliance	Monk	6	4	32	81208	28129	298	\N	1	dps	\N
4967	WG	WG40	Horde	Druid	0	3	29	53924	34405	375	1	\N	dps	\N
4968	WG	WG41	Alliance	Druid	1	5	14	49179	4133	309	\N	1	dps	1
4969	WG	WG41	Alliance	Druid	0	5	13	15325	0	301	\N	1	dps	1
4970	WG	WG41	Horde	Paladin	1	2	36	5279	95280	709	1	\N	heal	1
4971	WG	WG41	Alliance	Rogue	6	2	17	49015	4222	319	\N	1	dps	1
4972	WG	WG41	Horde	Priest	3	1	40	41963	114000	490	1	\N	heal	1
4973	WG	WG41	Alliance	Druid	0	6	6	31415	21123	276	\N	1	dps	1
4974	WG	WG41	Horde	Rogue	7	0	41	34279	5541	491	1	\N	dps	1
4975	WG	WG41	Alliance	Priest	1	6	16	19682	44098	313	\N	1	heal	1
4976	WG	WG41	Horde	Warlock	6	2	33	86577	49849	700	1	\N	dps	1
4977	WG	WG41	Alliance	Shaman	0	7	9	10888	57664	283	\N	1	heal	1
4978	WG	WG41	Alliance	Mage	4	2	13	58848	4638	131	\N	1	dps	1
4979	WG	WG41	Horde	Shaman	7	3	38	70954	4853	485	1	\N	dps	1
4980	WG	WG41	Horde	Demon Hunter	10	1	41	65618	14148	496	1	\N	dps	1
4981	WG	WG41	Horde	Druid	2	4	36	33003	20448	182	1	\N	dps	1
4982	WG	WG41	Horde	Hunter	3	4	34	35091	15295	706	1	\N	dps	1
4983	WG	WG41	Alliance	Paladin	1	4	12	66141	15367	302	\N	1	dps	1
4984	WG	WG41	Alliance	Paladin	3	5	16	106000	28499	314	\N	1	dps	1
4985	WG	WG41	Horde	Rogue	5	2	40	70420	18065	488	1	\N	dps	1
4986	WG	WG41	Horde	Warrior	4	1	34	25700	8531	705	1	\N	dps	1
4987	WG	WG41	Alliance	Hunter	4	6	11	61605	15730	294	\N	1	dps	1
4988	WG	WG42	Horde	Death Knight	7	3	39	79710	9567	530	1	\N	dps	\N
4989	WG	WG42	Alliance	Mage	4	5	44	84333	4704	278	\N	1	dps	\N
4990	WG	WG42	Alliance	Warrior	2	6	28	36170	7606	332	\N	1	dps	\N
4991	WG	WG42	Alliance	Priest	0	3	44	16260	150000	390	\N	1	heal	\N
4992	WG	WG42	Alliance	Priest	17	3	43	183000	38293	387	\N	1	dps	\N
4993	WG	WG42	Horde	Mage	5	5	42	78180	24147	382	1	\N	dps	\N
4994	WG	WG42	Horde	Hunter	4	3	42	104000	7376	533	1	\N	dps	\N
4995	WG	WG42	Horde	Paladin	23	3	46	158000	32676	392	1	\N	dps	\N
4996	WG	WG42	Alliance	Rogue	1	6	24	43471	3133	330	\N	1	dps	\N
4997	WG	WG42	Horde	Shaman	0	5	35	10648	72337	521	1	\N	heal	\N
4998	WG	WG42	Horde	Paladin	2	6	41	105000	27970	531	1	\N	dps	\N
4999	WG	WG42	Alliance	Demon Hunter	2	3	38	38418	2815	370	\N	1	dps	\N
5000	WG	WG42	Horde	Shaman	0	3	36	2385	42075	373	1	\N	heal	\N
5001	WG	WG42	Horde	Paladin	3	7	40	75692	17215	528	1	\N	dps	\N
5002	WG	WG42	Horde	Paladin	4	5	37	75061	30507	373	1	\N	dps	\N
5003	WG	WG42	Alliance	Warrior	6	6	43	82892	14101	387	\N	1	dps	\N
5004	WG	WG42	Alliance	Demon Hunter	1	5	36	45938	7169	366	\N	1	dps	\N
5005	WG	WG42	Horde	Druid	0	5	42	3662	203000	383	1	\N	heal	\N
5006	WG	WG42	Alliance	Paladin	0	4	38	12342	153000	375	\N	1	heal	\N
5007	WG	WG42	Alliance	Death Knight	10	6	40	126000	33219	383	\N	1	dps	\N
5008	SM	SM32	Alliance	Mage	2	3	22	56122	6572	767	1	\N	dps	\N
5009	SM	SM32	Alliance	Death Knight	2	0	20	54619	7493	535	1	\N	dps	\N
5010	SM	SM32	Alliance	Warlock	5	0	26	100000	66651	556	1	\N	dps	\N
5011	SM	SM32	Horde	Druid	2	4	16	83874	36822	252	\N	1	dps	\N
5012	SM	SM32	Horde	Druid	3	1	19	41207	5421	259	\N	1	dps	\N
5013	SM	SM32	Horde	Warrior	2	4	18	59347	8959	256	\N	1	dps	\N
5014	SM	SM32	Horde	Shaman	1	4	20	21032	149000	260	\N	1	heal	\N
5015	SM	SM32	Alliance	Paladin	1	5	22	104000	22181	769	1	\N	dps	\N
5016	SM	SM32	Alliance	Monk	2	5	21	50837	36765	758	1	\N	dps	\N
5017	SM	SM32	Horde	Warlock	1	3	16	69104	21543	251	\N	1	dps	\N
5018	SM	SM32	Horde	Monk	1	2	16	4785	114000	252	\N	1	heal	\N
5019	SM	SM32	Alliance	Shaman	2	2	17	113000	8504	521	1	\N	dps	\N
5020	SM	SM32	Alliance	Monk	1	0	23	8015	175000	766	1	\N	heal	\N
5021	SM	SM32	Alliance	Priest	3	2	18	21244	145000	523	1	\N	heal	\N
5022	SM	SM32	Alliance	Rogue	8	3	22	100000	13057	761	1	\N	dps	\N
5023	SM	SM32	Horde	Priest	1	2	17	23351	70766	213	\N	1	heal	\N
5024	SM	SM32	Horde	Paladin	2	4	18	106000	21396	256	\N	1	dps	\N
5025	SM	SM32	Alliance	Hunter	1	2	18	67814	14722	748	1	\N	dps	\N
5026	SM	SM32	Horde	Warlock	0	3	15	76533	48937	209	\N	1	dps	\N
5027	SM	SM32	Horde	Priest	9	0	18	112000	24350	216	\N	1	dps	\N
5028	SM	SM33	Alliance	Paladin	2	1	35	27577	36221	795	1	\N	heal	\N
5029	SM	SM33	Alliance	Hunter	5	3	35	55725	5588	792	1	\N	dps	\N
5030	SM	SM33	Alliance	Druid	0	1	37	98	187000	797	1	\N	heal	\N
5031	SM	SM33	Horde	Shaman	4	4	12	76776	16654	186	\N	1	dps	\N
5032	SM	SM33	Alliance	Druid	2	4	33	44876	7810	560	1	\N	dps	\N
5033	SM	SM33	Alliance	Priest	3	2	29	22680	10562	776	1	\N	dps	\N
5034	SM	SM33	Horde	Druid	0	4	12	43633	14753	181	\N	1	dps	\N
5035	SM	SM33	Alliance	Hunter	7	2	35	38255	16370	792	1	\N	dps	\N
5036	SM	SM33	Alliance	Mage	11	0	39	92521	12660	808	1	\N	dps	\N
5037	SM	SM33	Horde	Shaman	0	6	11	8073	77698	177	\N	1	heal	\N
5038	SM	SM33	Horde	Shaman	1	5	10	24224	6901	176	\N	1	dps	\N
5039	SM	SM33	Alliance	Druid	1	3	32	43973	7924	781	1	\N	dps	\N
5040	SM	SM33	Alliance	Hunter	3	2	34	46805	7971	798	1	\N	dps	\N
5041	SM	SM33	Horde	Hunter	0	1	3	995	3924	101	\N	1	dps	\N
5042	SM	SM33	Horde	Warlock	2	4	16	78083	13490	194	\N	1	dps	\N
5043	SM	SM33	Horde	Demon Hunter	2	3	12	44596	18643	185	\N	1	dps	\N
5044	SM	SM33	Horde	Priest	7	4	14	58066	23753	184	\N	1	dps	\N
5045	SM	SM33	Alliance	Mage	8	0	38	51209	7234	801	1	\N	dps	\N
5046	SM	SM33	Horde	Druid	0	5	12	19817	9927	180	\N	1	dps	\N
5047	SM	SM33	Horde	Priest	2	6	12	47490	18527	180	\N	1	dps	\N
5048	SM	SM34	Horde	Demon Hunter	0	0	35	2554	235	215	1	\N	dps	\N
5049	SM	SM34	Horde	Druid	3	0	37	50134	5833	519	1	\N	dps	\N
5050	SM	SM34	Horde	Monk	0	0	36	1437	68535	517	1	\N	heal	\N
5051	SM	SM34	Alliance	Warrior	0	6	3	25868	3136	195	\N	1	dps	\N
5052	SM	SM34	Horde	Shaman	1	0	34	13069	66975	513	1	\N	heal	\N
5053	SM	SM34	Alliance	Warrior	0	7	3	9007	0	194	\N	1	dps	\N
5054	SM	SM34	Horde	Warrior	6	1	34	45365	10790	513	1	\N	dps	\N
5055	SM	SM34	Horde	Demon Hunter	8	2	35	59124	17514	365	1	\N	dps	\N
5056	SM	SM34	Horde	Warlock	3	0	35	20757	8677	215	1	\N	dps	\N
5057	SM	SM34	Alliance	Mage	0	6	3	47598	25253	195	\N	1	dps	\N
5058	SM	SM34	Alliance	Death Knight	0	5	4	38473	18078	197	\N	1	dps	\N
5059	SM	SM34	Alliance	Paladin	1	0	6	11662	6629	205	\N	1	dps	\N
5060	SM	SM34	Horde	Paladin	6	1	36	63180	16043	517	1	\N	dps	\N
5061	SM	SM34	Horde	Demon Hunter	3	1	37	64251	3200	369	1	\N	dps	\N
5062	SM	SM34	Alliance	Hunter	1	1	6	32453	5316	205	\N	1	dps	\N
5063	SM	SM34	Alliance	Warrior	1	6	4	35014	6684	198	\N	1	dps	\N
5064	SM	SM34	Alliance	Priest	0	5	4	1180	30776	199	\N	1	heal	\N
5065	SM	SM34	Horde	Demon Hunter	4	1	33	48417	4708	361	1	\N	dps	\N
5066	SM	SM34	Alliance	Rogue	2	0	6	24701	5718	205	\N	1	dps	\N
5067	SM	SM35	Horde	Death Knight	5	5	22	63134	15379	216	\N	1	dps	\N
5068	SM	SM35	Alliance	Warrior	6	4	45	39943	16791	798	1	\N	dps	\N
5069	SM	SM35	Alliance	Mage	4	4	45	51297	15214	809	1	\N	dps	\N
5070	SM	SM35	Horde	Paladin	3	8	19	61903	19506	210	\N	1	dps	\N
5071	SM	SM35	Alliance	Paladin	5	1	52	81676	15527	826	1	\N	dps	\N
5072	SM	SM35	Alliance	Mage	3	0	52	47969	10457	600	1	\N	dps	\N
5073	SM	SM35	Alliance	Druid	0	4	48	802	119000	813	1	\N	heal	\N
5074	SM	SM35	Horde	Monk	2	6	20	31897	35895	212	\N	1	heal	\N
5075	SM	SM35	Alliance	Rogue	3	2	50	27941	16193	820	1	\N	dps	\N
5076	SM	SM35	Alliance	Warlock	3	2	49	39453	19576	595	1	\N	dps	\N
5077	SM	SM35	Horde	Rogue	3	6	18	33529	4200	208	\N	1	dps	\N
5078	SM	SM35	Alliance	Hunter	3	3	49	47121	2956	822	1	\N	dps	\N
5079	SM	SM35	Horde	Warlock	6	5	19	107000	44557	210	\N	1	dps	\N
5080	SM	SM35	Horde	Hunter	1	5	21	22950	6137	214	\N	1	dps	\N
5081	SM	SM35	Horde	Death Knight	1	5	19	58529	24431	211	\N	1	dps	\N
5082	SM	SM35	Horde	Paladin	1	3	18	31347	28833	207	\N	1	dps	\N
5083	SM	SM35	Alliance	Paladin	12	3	49	83020	35703	589	1	\N	dps	\N
5084	SM	SM35	Horde	Shaman	1	8	18	30583	7791	209	\N	1	dps	\N
5085	SM	SM35	Alliance	Warlock	15	1	53	88697	47874	824	1	\N	dps	\N
5086	SS	SS9	Alliance	Mage	3	3	26	54952	6762	992	1	\N	dps	\N
5087	SS	SS9	Horde	Mage	1	3	13	24599	10037	230	\N	1	dps	\N
5088	SS	SS9	Horde	Monk	0	7	9	35991	23694	222	\N	1	dps	\N
5089	SS	SS9	Horde	Druid	2	2	14	20864	11127	230	\N	1	dps	\N
5090	SS	SS9	Alliance	Mage	5	0	34	46508	7367	796	1	\N	dps	\N
5091	SS	SS9	Alliance	Rogue	5	1	27	22013	4940	999	1	\N	dps	\N
5092	SS	SS9	Alliance	Demon Hunter	5	2	34	43310	6727	1025	1	\N	dps	\N
5093	SS	SS9	Horde	Shaman	1	5	12	14220	60999	226	\N	1	heal	\N
5094	SS	SS9	Horde	Priest	2	3	14	32813	25281	230	\N	1	dps	\N
5095	SS	SS9	Alliance	Death Knight	7	2	34	75995	5808	1025	1	\N	dps	\N
5096	SS	SS9	Horde	Death Knight	4	3	15	64440	29149	234	\N	1	dps	\N
5097	SS	SS9	Horde	Monk	1	3	14	1451	31376	232	\N	1	heal	\N
5098	SS	SS9	Alliance	Rogue	0	4	14	25166	3935	234	1	\N	dps	\N
5099	SS	SS9	Horde	Warrior	2	4	14	25166	3935	234	\N	1	dps	\N
5100	SS	SS9	Alliance	Shaman	6	1	28	60491	2870	547	1	\N	dps	\N
5101	SS	SS9	Alliance	Monk	1	2	32	3159	82580	792	1	\N	heal	\N
5102	SS	SS9	Alliance	Priest	1	1	33	20648	77266	793	1	\N	heal	\N
5103	SS	SS9	Horde	Demon Hunter	2	3	14	57090	6987	236	\N	1	dps	\N
5104	SS	SS9	Horde	Warlock	2	4	12	22342	13107	228	\N	1	dps	\N
5105	SS	SS9	Alliance	Warlock	4	1	32	26007	8656	1015	1	\N	dps	\N
5106	SS	SS10	Alliance	Hunter	2	4	7	16596	5073	357	\N	1	dps	\N
5107	SS	SS10	Alliance	Shaman	0	5	7	9549	52229	359	\N	1	heal	\N
5108	SS	SS10	Horde	Paladin	0	1	32	7508	67367	515	1	\N	heal	\N
5109	SS	SS10	Horde	Hunter	5	0	29	47539	5520	517	1	\N	dps	\N
5110	SS	SS10	Horde	Shaman	0	3	36	6901	78230	673	1	\N	heal	\N
5111	SS	SS10	Horde	Warlock	6	0	29	40412	13959	367	1	\N	dps	\N
5112	SS	SS10	Alliance	Warlock	0	6	8	56324	24242	362	\N	1	dps	\N
5113	SS	SS10	Alliance	Hunter	0	1	8	881	3638	243	\N	1	dps	\N
5114	SS	SS10	Horde	Mage	5	1	36	54309	2173	523	1	\N	dps	\N
5115	SS	SS10	Horde	Rogue	3	2	28	43543	6544	657	1	\N	dps	\N
5116	SS	SS10	Alliance	Paladin	1	5	7	36680	18033	357	\N	1	dps	\N
5117	SS	SS10	Alliance	Paladin	0	1	0	10552	366	127	\N	1	dps	\N
5118	SS	SS10	Horde	Hunter	2	1	35	44508	7120	671	1	\N	dps	\N
5119	SS	SS10	Alliance	Druid	0	4	0	4788	44189	225	\N	1	heal	\N
5120	SS	SS10	Horde	Hunter	2	0	37	20784	3783	529	1	\N	dps	\N
5121	SS	SS10	Horde	Hunter	8	0	38	74877	5119	531	1	\N	dps	\N
5122	SS	SS10	Alliance	Druid	1	5	7	26039	9302	357	\N	1	dps	\N
5123	SS	SS10	Horde	Warlock	7	0	38	67584	8346	681	1	\N	dps	\N
5124	SS	SS10	Alliance	Priest	1	5	2	17764	9891	344	\N	1	dps	\N
5125	SS	SS10	Alliance	Warrior	1	4	7	30250	11659	245	\N	1	dps	\N
5126	SS	SS11	Horde	Rogue	7	1	23	64094	19694	553	1	\N	dps	1
5127	SS	SS11	Alliance	Mage	2	3	26	12507	3022	477	\N	1	dps	1
5128	SS	SS11	Alliance	Mage	3	4	21	51608	15021	632	\N	1	dps	1
5129	SS	SS11	Alliance	Warlock	4	2	32	54580	15617	496	\N	1	dps	1
5130	SS	SS11	Alliance	Priest	0	2	32	11796	65218	664	\N	1	heal	1
5131	SS	SS11	Alliance	Hunter	0	2	17	14429	3416	621	\N	1	dps	1
5132	SS	SS11	Horde	Shaman	0	6	21	15812	99577	812	1	\N	heal	1
5133	SS	SS11	Horde	Warrior	4	3	14	48480	12229	799	1	\N	dps	1
5134	SS	SS11	Horde	Hunter	4	4	18	42095	10556	350	1	\N	dps	1
5135	SS	SS11	Alliance	Druid	5	2	31	69690	23319	661	\N	1	dps	1
5136	SS	SS11	Horde	Priest	1	3	17	18837	73922	803	1	\N	heal	1
5137	SS	SS11	Horde	Demon Hunter	0	5	20	71121	9991	809	1	\N	dps	1
5138	SS	SS11	Alliance	Priest	0	2	30	14976	133	656	\N	1	dps	1
5139	SS	SS11	Horde	Hunter	0	3	25	33551	7821	598	1	\N	dps	1
5140	SS	SS11	Horde	Mage	1	3	17	16657	13145	290	1	\N	dps	1
5141	SS	SS11	Horde	Hunter	1	5	22	33794	5381	589	1	\N	dps	1
5142	SS	SS11	Alliance	Mage	7	2	33	114000	17315	666	\N	1	dps	1
5143	SS	SS11	Horde	Demon Hunter	8	1	23	62754	17580	360	1	\N	dps	1
5144	SS	SS11	Alliance	Warlock	8	3	32	95514	29302	666	\N	1	dps	1
5145	SS	SS11	Alliance	Rogue	6	4	29	31916	5341	654	\N	1	dps	1
5146	SA	SA8	Horde	Shaman	1	5	28	16505	21669	190	1	\N	heal	\N
5147	SA	SA8	Horde	Warlock	4	8	28	156000	80823	489	1	\N	dps	\N
5148	SA	SA8	Horde	Rogue	4	2	31	72742	26661	493	1	\N	dps	\N
5149	SA	SA8	Horde	Mage	1	4	29	78128	36980	486	1	\N	dps	\N
5150	SA	SA8	Horde	Warlock	1	3	25	135000	58611	266	1	\N	dps	\N
5151	SA	SA8	Alliance	Death Knight	5	5	50	96727	41737	350	\N	1	dps	\N
5152	SA	SA8	Alliance	Warlock	8	2	55	127000	70339	369	\N	1	dps	\N
5153	SA	SA8	Alliance	Warlock	2	3	49	48113	39407	341	\N	1	dps	\N
5154	SA	SA8	Horde	Warlock	1	5	23	58579	36711	277	1	\N	dps	\N
5155	SA	SA8	Horde	Shaman	0	3	26	15314	103000	274	1	\N	heal	\N
5156	SA	SA8	Alliance	Hunter	4	3	54	153000	17123	341	\N	1	dps	\N
5157	SA	SA8	Alliance	Monk	0	1	56	5971	191000	369	\N	1	heal	\N
5158	SA	SA8	Alliance	Demon Hunter	3	3	41	52632	10972	305	\N	1	dps	\N
5159	SA	SA8	Horde	Hunter	2	6	28	34133	11469	485	1	\N	dps	\N
5160	SA	SA8	Alliance	Paladin	5	3	50	63002	17084	346	\N	1	dps	\N
5161	SA	SA8	Horde	Monk	1	2	26	2594	49773	265	1	\N	heal	\N
5162	SA	SA8	Horde	Mage	1	9	27	78068	25887	488	1	\N	dps	\N
5163	SA	SA8	Alliance	Paladin	6	3	47	79330	20159	342	\N	1	dps	\N
5164	SA	SA8	Alliance	Priest	8	1	60	123000	30672	362	\N	1	dps	\N
5165	SA	SA8	Alliance	Rogue	3	4	56	49321	10935	347	\N	1	dps	\N
5166	SA	SA8	Alliance	Paladin	3	1	43	20760	17811	311	\N	1	dps	\N
5167	SA	SA8	Horde	Warlock	4	5	26	115000	66300	335	1	\N	dps	\N
5168	SA	SA8	Alliance	Priest	12	3	62	212000	52515	383	\N	1	dps	\N
5169	SA	SA8	Alliance	Warlock	8	0	60	109000	47507	361	\N	1	dps	\N
5170	SA	SA8	Horde	Druid	0	7	22	21203	15743	475	1	\N	dps	\N
5171	SA	SA8	Alliance	Paladin	0	1	52	15401	201000	352	\N	1	heal	\N
5172	SA	SA8	Horde	Warlock	5	4	28	129000	51885	357	1	\N	dps	\N
5173	SA	SA8	Horde	Paladin	1	5	29	6980	159000	345	1	\N	heal	\N
5174	SA	SA8	Alliance	Warrior	3	2	50	47300	11999	383	\N	1	dps	\N
5175	SA	SA8	Horde	Shaman	10	1	31	124000	20049	497	1	\N	dps	\N
5176	SA	SA9	Alliance	Death Knight	5	3	39	111000	25365	218	1	\N	dps	\N
5177	SA	SA9	Alliance	Paladin	3	4	50	94213	19768	568	1	\N	dps	\N
5178	SA	SA9	Horde	Druid	1	3	70	15470	309000	266	\N	1	heal	\N
5179	SA	SA9	Horde	Priest	4	2	68	67728	12948	267	\N	1	dps	\N
5180	SA	SA9	Alliance	Mage	0	9	50	91243	12241	332	1	\N	dps	\N
5181	SA	SA9	Alliance	Druid	1	5	49	14854	139000	575	1	\N	heal	\N
5182	SA	SA9	Alliance	Druid	3	10	51	92835	18101	325	1	\N	dps	\N
5183	SA	SA9	Alliance	Priest	2	6	50	43033	231000	553	1	\N	heal	\N
5184	SA	SA9	Horde	Mage	1	2	69	191000	28581	261	\N	1	dps	\N
5185	SA	SA9	Horde	Shaman	0	5	67	39058	147000	258	\N	1	heal	\N
5186	SA	SA9	Alliance	Rogue	1	5	55	48795	12820	339	1	\N	dps	\N
5187	SA	SA9	Horde	Paladin	13	5	65	161000	53241	264	\N	1	dps	\N
5188	SA	SA9	Horde	Monk	8	3	68	104000	34185	259	\N	1	dps	\N
5189	SA	SA9	Alliance	Rogue	4	7	49	58295	26393	772	1	\N	dps	\N
5190	SA	SA9	Horde	Warlock	7	4	68	120000	36376	260	\N	1	dps	\N
5191	SA	SA9	Alliance	Rogue	2	2	48	36811	6773	342	1	\N	dps	\N
5192	SA	SA9	Horde	Druid	1	2	47	40849	67973	243	\N	1	heal	\N
5193	SA	SA9	Alliance	Warlock	8	6	56	170000	44020	336	1	\N	dps	\N
5194	SA	SA9	Horde	Druid	0	0	13	15318	54980	129	\N	1	heal	\N
5195	SA	SA9	Alliance	Druid	1	2	60	4805	209000	348	1	\N	heal	\N
5196	SA	SA9	Horde	Rogue	6	5	68	105000	22143	263	\N	1	dps	\N
5197	SA	SA9	Alliance	Warlock	5	7	44	151000	99119	780	1	\N	dps	\N
5198	SA	SA9	Horde	Mage	13	6	67	159000	16895	255	\N	1	dps	\N
5199	SA	SA9	Horde	Warlock	6	4	64	82821	38752	254	\N	1	dps	\N
5200	SA	SA9	Alliance	Rogue	8	4	60	71319	12960	587	1	\N	dps	\N
5201	SA	SA9	Alliance	Warlock	7	6	52	156000	80176	813	1	\N	dps	\N
5202	SA	SA9	Horde	Death Knight	4	7	63	99656	15526	253	\N	1	dps	\N
5203	SA	SA9	Alliance	Death Knight	14	5	59	186000	63877	800	1	\N	dps	\N
5204	SA	SA9	Horde	Warrior	9	3	74	171000	32828	271	\N	1	dps	\N
5205	SA	SA9	Horde	Shaman	6	10	62	96791	8124	259	\N	1	dps	\N
5206	SM	SM36	Alliance	Druid	2	1	48	9015	119000	707	1	\N	heal	1
5207	SM	SM36	Alliance	Hunter	3	2	47	26042	1165	699	1	\N	dps	1
5208	SM	SM36	Horde	Shaman	2	7	10	49913	14078	241	\N	1	dps	1
5209	SM	SM36	Alliance	Warrior	18	2	50	152000	10946	711	1	\N	dps	1
5210	SM	SM36	Horde	Warlock	0	7	6	37250	46128	233	\N	1	dps	1
5211	SM	SM36	Alliance	Hunter	7	0	49	82066	0	710	1	\N	dps	1
5212	SM	SM36	Horde	Death Knight	0	6	9	72513	30174	241	\N	1	dps	1
5213	SM	SM36	Horde	Rogue	4	4	8	55802	12003	243	\N	1	dps	1
5214	SM	SM36	Horde	Druid	1	6	5	68974	34774	231	\N	1	dps	1
5215	SM	SM36	Alliance	Demon Hunter	6	0	50	115000	6642	1054	1	\N	dps	1
5216	SM	SM36	Horde	Shaman	0	9	5	7297	96346	231	\N	1	heal	1
5217	SM	SM36	Alliance	Death Knight	2	2	46	40316	10293	697	1	\N	dps	1
5218	SM	SM36	Alliance	Priest	0	1	54	10454	167000	728	1	\N	heal	1
5219	SM	SM36	Alliance	Warlock	2	2	39	44548	16678	1018	1	\N	dps	1
5220	SM	SM36	Horde	Death Knight	1	7	9	49611	11374	239	\N	1	dps	1
5221	SM	SM36	Alliance	Priest	10	1	51	128000	15745	720	1	\N	dps	1
5222	SM	SM36	Horde	Rogue	1	2	10	18681	17023	247	\N	1	dps	1
5223	SM	SM36	Horde	Monk	0	7	10	293	95806	241	\N	1	heal	1
5224	SM	SM36	Horde	Warlock	0	1	0	28679	10798	182	\N	1	dps	1
5225	SM	SM36	Alliance	Death Knight	6	0	43	80985	11593	703	1	\N	dps	1
5226	SM	SM37	Horde	Hunter	2	0	27	56290	4098	365	1	\N	dps	\N
5227	SM	SM37	Alliance	Hunter	1	2	2	16054	6887	162	\N	1	dps	\N
5228	SM	SM37	Horde	Priest	0	0	26	7282	30631	364	1	\N	heal	\N
5229	SM	SM37	Horde	Druid	0	0	21	17190	1667	353	1	\N	dps	\N
5230	SM	SM37	Alliance	Monk	0	2	2	579	79388	162	\N	1	heal	\N
5231	SM	SM37	Horde	Shaman	1	0	24	3707	37110	359	1	\N	heal	\N
5232	SM	SM37	Horde	Shaman	3	0	18	28779	2100	307	1	\N	dps	\N
5233	SM	SM37	Horde	Druid	2	1	25	26305	16785	362	1	\N	dps	\N
5234	SM	SM37	Alliance	Priest	0	6	2	13599	2667	162	\N	1	dps	\N
5235	SM	SM37	Horde	Mage	4	0	27	33880	6620	515	1	\N	dps	\N
5236	SM	SM37	Horde	Hunter	0	2	24	7881	5981	360	1	\N	dps	\N
5237	SM	SM37	Horde	Priest	12	0	28	38918	5733	368	1	\N	dps	\N
5238	SM	SM37	Alliance	Shaman	0	2	2	45410	7341	162	\N	1	dps	\N
5239	SM	SM37	Alliance	Demon Hunter	0	1	0	3309	0	157	\N	1	dps	\N
5240	SM	SM37	Alliance	Hunter	1	2	2	8205	2816	132	\N	1	dps	\N
5241	SM	SM37	Alliance	Mage	0	4	3	23127	4090	167	\N	1	dps	\N
5242	SM	SM37	Alliance	Druid	0	2	2	7682	3543	134	\N	1	dps	\N
5243	SM	SM37	Horde	Druid	3	0	24	33961	6901	510	1	\N	dps	\N
5244	SM	SM37	Alliance	Priest	1	5	3	17254	3347	167	\N	1	dps	\N
5245	SM	SM37	Alliance	Druid	0	2	3	3502	2019	167	\N	1	dps	\N
5246	SS	SS12	Alliance	Mage	0	2	13	30190	10725	852	1	\N	dps	1
5247	SS	SS12	Horde	Warlock	6	0	28	53941	9111	337	\N	1	dps	1
5248	SS	SS12	Alliance	Hunter	2	3	14	32413	12528	859	1	\N	dps	1
5249	SS	SS12	Alliance	Paladin	0	4	10	31374	11616	843	1	\N	dps	1
5250	SS	SS12	Horde	Rogue	6	0	28	50923	3591	332	\N	1	dps	1
5251	SS	SS12	Alliance	Mage	3	2	13	21152	3465	853	1	\N	dps	1
5252	SS	SS12	Horde	Mage	4	1	26	36130	9593	335	\N	1	dps	1
5253	SS	SS12	Alliance	Priest	0	4	9	5048	55868	1177	1	\N	heal	1
5254	SS	SS12	Horde	Warlock	1	2	24	27830	13165	329	\N	1	dps	1
5255	SS	SS12	Alliance	Hunter	1	3	14	28185	4515	858	1	\N	dps	1
5256	SS	SS12	Alliance	Warlock	2	1	14	44672	33148	859	1	\N	dps	1
5257	SS	SS12	Horde	Death Knight	6	3	22	50774	15880	331	\N	1	dps	1
5258	SS	SS12	Horde	Hunter	1	4	17	42340	19107	316	\N	1	dps	1
5259	SS	SS12	Alliance	Paladin	5	4	11	56078	21272	1183	1	\N	dps	1
5260	SS	SS12	Horde	Shaman	1	1	28	21378	13303	339	\N	1	dps	1
5261	SS	SS12	Horde	Mage	2	2	24	32493	14173	329	\N	1	dps	1
5262	SS	SS12	Alliance	Druid	0	4	12	455	36879	1187	1	\N	heal	1
5263	SS	SS12	Horde	Shaman	1	3	27	7876	84232	336	\N	1	heal	1
5264	SS	SS12	Alliance	Mage	2	2	14	18878	9055	861	1	\N	dps	1
5265	SS	SS12	Horde	Paladin	0	0	14	12418	7381	249	\N	1	dps	1
5266	SS	SS13	Horde	Paladin	9	2	30	187000	48454	374	1	\N	dps	1
5267	SS	SS13	Alliance	Warlock	7	4	26	59204	33687	440	\N	1	dps	1
5268	SS	SS13	Alliance	Priest	0	2	22	14198	52627	353	\N	1	heal	1
5269	SS	SS13	Horde	Demon Hunter	5	0	28	126000	2302	818	1	\N	dps	1
5270	SS	SS13	Alliance	Warrior	3	5	13	46585	11788	468	\N	1	dps	1
5271	SS	SS13	Horde	Druid	0	3	19	60804	15099	801	1	\N	dps	1
5272	SS	SS13	Alliance	Warrior	1	3	26	97875	25607	614	\N	1	dps	1
5273	SS	SS13	Alliance	Warlock	6	3	21	64252	27808	598	\N	1	dps	1
5274	SS	SS13	Horde	Demon Hunter	5	4	27	82074	12799	818	1	\N	dps	1
5275	SS	SS13	Horde	Shaman	0	3	27	35999	108000	816	1	\N	heal	1
5276	SS	SS13	Horde	Death Knight	3	5	18	81760	18891	803	1	\N	dps	1
5277	SS	SS13	Alliance	Druid	0	5	25	25499	208000	606	\N	1	heal	1
5278	SS	SS13	Alliance	Warrior	1	3	25	32515	14308	605	\N	1	dps	1
5279	SS	SS13	Alliance	Paladin	6	3	27	103000	26801	444	\N	1	dps	1
5280	SS	SS13	Horde	Monk	1	1	28	9178	150000	593	1	\N	heal	1
5281	SS	SS13	Alliance	Shaman	2	6	22	9050	152000	598	\N	1	heal	1
5282	SS	SS13	Horde	Demon Hunter	3	4	27	122000	5594	817	1	\N	dps	1
5283	SS	SS13	Alliance	Priest	3	2	28	54424	136000	616	\N	1	heal	1
5284	SS	SS13	Horde	Shaman	3	5	22	119000	8537	366	1	\N	dps	1
5285	SS	SS13	Horde	Rogue	2	3	22	63721	9944	359	1	\N	dps	1
5286	SS	SS14	Horde	Demon Hunter	2	4	16	72209	4672	580	1	\N	dps	1
5287	SS	SS14	Alliance	Death Knight	0	2	6	24195	14461	498	\N	1	dps	1
5288	SS	SS14	Horde	Druid	1	2	20	38152	3819	585	1	\N	dps	1
5289	SS	SS14	Horde	Druid	1	1	25	12961	5536	595	1	\N	dps	1
5290	SS	SS14	Horde	Mage	8	2	23	74049	7391	820	1	\N	dps	1
5291	SS	SS14	Alliance	Paladin	0	3	15	6898	42265	528	\N	1	heal	1
5292	SS	SS14	Alliance	Druid	0	2	14	16120	7198	519	\N	1	dps	1
5293	SS	SS14	Horde	Shaman	0	5	15	9223	80831	574	1	\N	heal	1
5294	SS	SS14	Alliance	Priest	0	3	15	3115	108000	522	\N	1	heal	1
5295	SS	SS14	Alliance	Paladin	2	4	14	66243	13281	522	\N	1	dps	1
5296	SS	SS14	Alliance	Hunter	1	3	20	35144	5326	540	\N	1	dps	1
5297	SS	SS14	Alliance	Paladin	3	3	17	96068	19388	528	\N	1	dps	1
5298	SS	SS14	Alliance	Warrior	5	2	20	57879	12716	540	\N	1	dps	1
5299	SS	SS14	Alliance	Paladin	4	1	20	55849	18951	540	\N	1	dps	1
5300	SS	SS14	Horde	Druid	0	1	24	3928	110000	369	1	\N	heal	1
5301	SS	SS14	Alliance	Warlock	5	2	20	57497	26200	540	\N	1	dps	1
5302	SS	SS14	Horde	Rogue	5	2	17	31496	4087	800	1	\N	dps	1
5303	SS	SS14	Horde	Warrior	1	4	18	28592	48381	805	1	\N	dps	1
5304	SS	SS14	Horde	Mage	3	0	24	57301	3914	815	1	\N	dps	1
5305	SS	SS14	Horde	Paladin	4	1	24	81572	20519	594	1	\N	dps	1
5306	TK	TK37	Alliance	Paladin	0	3	3	555	44219	236	\N	1	heal	1
5307	TK	TK37	Alliance	Mage	1	3	5	44853	8256	242	\N	1	dps	1
5308	TK	TK37	Horde	Mage	1	1	27	15709	4957	651	1	\N	dps	1
5309	TK	TK37	Alliance	Druid	0	5	4	29216	2810	240	\N	1	dps	1
5310	TK	TK37	Horde	Demon Hunter	2	0	30	28427	2332	432	1	\N	dps	1
5311	TK	TK37	Horde	Druid	1	2	25	27448	4418	647	1	\N	dps	1
5312	TK	TK37	Horde	Shaman	0	1	30	10822	60909	432	1	\N	heal	1
5313	TK	TK37	Alliance	Shaman	0	1	5	798	54216	242	\N	1	heal	1
5314	TK	TK37	Horde	Druid	1	0	30	2447	81635	657	1	\N	heal	1
5315	TK	TK37	Alliance	Paladin	1	0	5	9173	1818	242	\N	1	dps	1
5316	TK	TK37	Horde	Warlock	5	0	30	31319	9720	657	1	\N	dps	1
5317	TK	TK37	Alliance	Rogue	0	4	5	25900	4187	242	\N	1	dps	1
5318	TK	TK37	Alliance	Paladin	0	1	3	1457	1294	236	\N	1	dps	1
5319	TK	TK37	Alliance	Paladin	3	4	4	25381	17957	239	\N	1	dps	1
5320	TK	TK37	Alliance	Warrior	0	5	2	23667	3103	233	\N	1	dps	1
5321	TK	TK37	Alliance	Warrior	0	4	5	41483	5582	242	\N	1	dps	1
5322	TK	TK37	Horde	Death Knight	4	0	30	35696	3410	657	1	\N	dps	1
5323	TK	TK37	Horde	Priest	9	0	30	68485	5179	657	1	\N	dps	1
5324	TK	TK37	Horde	Mage	4	0	30	57384	4506	432	1	\N	dps	1
5325	TK	TK37	Horde	Shaman	2	1	26	30112	4028	199	1	\N	dps	1
5326	TK	TK38	Alliance	Mage	0	4	6	17171	3956	260	\N	1	dps	1
5327	TK	TK38	Alliance	Rogue	0	2	6	11000	8347	260	\N	1	dps	1
5328	TK	TK38	Alliance	Mage	0	3	5	36622	14091	257	\N	1	dps	1
5329	TK	TK38	Horde	Mage	2	0	41	34234	7101	448	1	\N	dps	1
5330	TK	TK38	Horde	Priest	6	0	41	33525	4915	448	1	\N	dps	1
5331	TK	TK38	Alliance	Priest	0	4	6	21221	6169	260	\N	1	dps	1
5332	TK	TK38	Alliance	Shaman	1	3	3	10452	7745	221	\N	1	dps	1
5333	TK	TK38	Horde	Paladin	0	0	41	427	61994	448	1	\N	heal	1
5334	TK	TK38	Horde	Priest	9	1	40	80495	19072	446	1	\N	dps	1
5335	TK	TK38	Horde	Hunter	4	3	36	42145	10047	663	1	\N	dps	1
5336	TK	TK38	Alliance	Monk	0	4	3	0	81867	257	\N	1	heal	1
5337	TK	TK38	Alliance	Demon Hunter	1	4	5	33820	3261	257	\N	1	dps	1
5338	TK	TK38	Horde	Shaman	0	0	41	2642	73680	448	1	\N	heal	1
5339	TK	TK38	Horde	Warrior	6	1	39	59775	7480	444	1	\N	dps	1
5340	TK	TK38	Horde	Druid	6	0	41	84645	12062	448	1	\N	dps	1
5341	TK	TK38	Alliance	Druid	0	5	2	9027	23956	248	\N	1	heal	1
5342	TK	TK38	Alliance	Mage	0	5	4	39006	14300	255	\N	1	dps	1
5343	TK	TK38	Alliance	Hunter	4	5	5	41808	9923	257	\N	1	dps	1
5344	TK	TK38	Horde	Paladin	5	0	41	35167	11716	673	1	\N	dps	1
5345	TK	TK38	Horde	Rogue	2	1	36	22197	1743	438	1	\N	dps	1
5346	TK	TK39	Horde	Death Knight	2	4	16	29815	20363	211	\N	1	dps	1
5347	TK	TK39	Horde	Death Knight	5	4	16	55496	12646	211	\N	1	dps	1
5348	TK	TK39	Horde	Rogue	3	5	17	64375	9229	213	\N	1	dps	1
5349	TK	TK39	Alliance	Hunter	2	0	45	36949	2762	1027	1	\N	dps	1
5350	TK	TK39	Alliance	Rogue	2	3	43	51363	6219	1021	1	\N	dps	1
5351	TK	TK39	Horde	Druid	0	5	11	37458	21341	201	\N	1	dps	1
5352	TK	TK39	Alliance	Warrior	10	1	43	58396	9107	685	1	\N	dps	1
5353	TK	TK39	Horde	Shaman	0	5	15	11775	102000	209	\N	1	heal	1
5354	TK	TK39	Horde	Priest	0	0	5	5385	31426	150	\N	1	heal	1
5355	TK	TK39	Horde	Demon Hunter	3	4	13	64724	6589	205	\N	1	dps	1
5356	TK	TK39	Alliance	Mage	0	3	37	46380	3792	672	1	\N	dps	1
5357	TK	TK39	Alliance	Druid	2	4	34	24950	10005	1001	1	\N	dps	1
5358	TK	TK39	Alliance	Paladin	0	0	45	9174	129000	690	1	\N	heal	1
5359	TK	TK39	Alliance	Paladin	13	1	45	214000	30619	690	1	\N	dps	1
5360	TK	TK39	Horde	Paladin	4	5	17	108000	31205	213	\N	1	dps	1
5361	TK	TK39	Alliance	Shaman	12	1	43	99841	5216	1021	1	\N	dps	1
5362	TK	TK39	Alliance	Priest	1	1	44	23353	142000	687	1	\N	heal	1
5363	TK	TK39	Alliance	Warrior	3	4	32	36094	5749	660	1	\N	dps	1
5364	TK	TK39	Horde	Demon Hunter	1	4	14	36223	7653	207	\N	1	dps	1
5365	TK	TK39	Horde	Druid	0	5	16	0	75726	211	\N	1	heal	1
5366	TK	TK40	Horde	Warrior	0	2	37	33725	2266	445	1	\N	dps	1
5367	TK	TK40	Alliance	Monk	0	5	10	0	27143	283	\N	1	heal	1
5368	TK	TK40	Horde	Warlock	7	1	47	58042	20170	690	1	\N	dps	1
5369	TK	TK40	Horde	Hunter	1	0	47	40931	1401	690	1	\N	dps	1
5370	TK	TK40	Horde	Paladin	8	1	46	76777	18438	463	1	\N	dps	1
5371	TK	TK40	Alliance	Death Knight	0	3	6	17290	17949	272	\N	1	dps	1
5372	TK	TK40	Horde	Shaman	1	2	42	16649	107000	680	1	\N	heal	1
5373	TK	TK40	Horde	Warlock	5	0	47	52466	5647	465	1	\N	dps	1
5374	TK	TK40	Horde	Demon Hunter	1	3	37	33043	1535	445	1	\N	dps	1
5375	TK	TK40	Alliance	Hunter	1	6	10	30219	6039	286	\N	1	dps	1
5376	TK	TK40	Horde	Death Knight	10	2	43	74966	21467	682	1	\N	dps	1
5377	TK	TK40	Horde	Hunter	6	1	45	108000	5539	461	1	\N	dps	1
5378	TK	TK40	Alliance	Druid	0	6	8	0	63814	278	\N	1	heal	1
5379	TK	TK40	Horde	Shaman	3	0	47	19154	2611	465	1	\N	dps	1
5380	TK	TK40	Alliance	Mage	1	4	12	26117	11492	291	\N	1	dps	1
5381	TK	TK40	Alliance	Demon Hunter	2	4	9	39954	9699	282	\N	1	dps	1
5382	TK	TK40	Alliance	Paladin	4	6	9	54729	20736	282	\N	1	dps	1
5383	TK	TK40	Alliance	Rogue	0	5	10	11495	20355	283	\N	1	dps	1
\.


--
-- Name: bgs_data_Id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public."bgs_data_Id_seq"', 5383, true);


--
-- Name: bg_statistic bgs_data_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bg_statistic
    ADD CONSTRAINT bgs_data_pkey PRIMARY KEY ("Id");


--
-- PostgreSQL database dump complete
--

