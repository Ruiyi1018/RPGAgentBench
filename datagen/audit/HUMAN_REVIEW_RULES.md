# Foundation、Profile与Anchor人工审核清单

目标不是检查“故事是否好看”，而是保证生成的数据能有效测试：

- **Stage 1**：NPC能否记住普通细节、事件顺序和多次变化后的当前状态。
- **Stage 2**：NPC能否只在关键状态改变时改变决定，并忽略不关键的表述变化。

## 1. 先区分三类材料

- **Anchor**：会改变知识、目标/承诺、关系、权限或资源状态的关键事件。
- **普通记忆**：不改变关键状态，但之后需要准确回忆的名字、时间、地点、物品或约定细节。
- **干扰信息**：与任务相似但不应覆盖正确记忆的信息；不得与正典事实自相矛盾。

不要为了“转折多”把所有对话都写成Anchor。Stage 1需要的是可区分、可追踪的
多次变化，以及跨Session的普通记忆，而不是连续戏剧冲突。

## 2. Foundation与Profile门禁

以下任一项不满足即退回：

1. 作品版本和`shared_snapshot`唯一明确，所有角色的身份、职务和存活状态同时成立。
2. Profile不含快照之后的剧情、未来身份或观众全知信息。
3. 公开事实、NPC私密知识、传闻和未知信息没有混写；Player默认不知道私密信息。
4. 权限、限制、关系和禁区能导出具体的允许/拒绝决定，而非抽象性格词。
5. Profile引用的地点、物品和组织均存在于世界资产中。
6. Reviewer能用一句话说明角色“现在是谁”，并指出至少一项角色此时不知道的事实。

## 3. Stage 1：记忆与转折审核

### 生成历史前：审核Anchor与Transition

每名Longitudinal角色使用30个Anchor，其中10个正典、20个受控事件。整组必须：

1. 覆盖至少12条不同状态路径，并覆盖知识、目标/承诺、关系、资源/权限四类状态。
2. 至少8个Anchor真正构成转折：撤销、履行、失败、替代、纠正或覆盖一个先前状态；
   不能30次都只是新增互不相关的信息。
3. 每次转折都能回答“旧状态是什么—什么证据触发—新状态是什么—何时开始生效”。
4. 同一状态多次变化时，后来的状态明确覆盖前者，但历史事实仍可被回忆。
5. 每个Anchor之后都有具体可观察决定，例如改变授权、核查渠道、合作对象或资源分配；
   “更谨慎”“更生气”不算后果。

### 生成历史后：审核600轮历史与QA

1. 数量固定为600轮、30个Session、30个Anchor、50道QA、2个开放任务和10组Pair。
2. `generation_report`中的`quality.history_value.passed`为`true`。
3. 至少6条普通记忆跨Session再次被使用，并实际形成6条`episodic` QA。
4. 普通记忆具体、唯一、可客观判分；相似干扰信息不得制造无解歧义。
5. QA配比为：Profile 6、时序4、普通记忆6、局部状态8、检查点状态8、
   最终状态12、多跳6。
6. 状态题同时测试变化刚发生后、中途检查点和600轮结束时的有效状态。
7. Anchor位置分散，包含无Anchor和多Anchor的Session，并包含延迟显现的后果。

Stage 1不通过的典型情况：只记剧情梗概；变化没有覆盖关系；关键事实提前出现；
600轮充满重复寒暄；普通记忆没有在后续Session被调用。

## 4. Stage 2：关键与不关键变化审核

“关键/不关键”描述的是**分支干预对最终决定是否应有影响**，不是把Anchor分成
“有用”和“无用”。所有入选Anchor都必须有状态价值。

每名角色必须恰好提供：

- **7个Sensitivity Pair**：只改一个关键原因或证据，导致状态和最终决定都应改变。
- **3个Invariance Pair**：只做同一事件的等价改写，状态和最终决定都不应改变。

逐个Pair检查：

1. 两分支可见历史只有目标干预不同。
2. Sensitivity的`cf_edit`只改一个原因，且两侧Operation产生不同状态。
3. 最终问题确实依赖该状态；不读历史时不能仅凭常识或角色Profile答对。
4. 两侧rubric要求明确不同的可观察决定，不只比较语气。
5. Invariance保持事实语义、状态和正确决定不变，不能暗中增删证据。
6. 关键干预、普通记忆和干扰信息彼此可区分，不让无关细节意外决定答案。

## 5. 每个Anchor的八问

全部回答“是”才通过：

1. 事件此前尚未发生，且时间顺序正确吗？
2. 角色确实能通过在场、被告知或证据获知它吗？
3. Player可见范围明确，隐藏事实没有被Player或NPC直接泄漏吗？
4. Operation由当时证据支持且可执行吗？
5. 新状态与前态的新增、覆盖、撤销或闭合关系明确吗？
6. 后续至少一次具体决定能观察到该变化吗？
7. 它为Stage 1局部/检查点/最终状态题或Stage 2 Pair提供明确用途吗？
8. 若用于Sensitivity，反事实是否只改变一个关键原因？若用于Invariance，
   改写是否保持语义等价？若`probe_type=none`，本项记为N/A。

一票否决：时间倒置、未来知识、隐藏信息直泄、无实际状态差异、生命周期断裂、
反事实多处变化、最终问题与目标状态无关。

## 6. 审核与冻结

顺序固定为Foundation/Profile → Anchor/Transition → 生成历史与Stage 1/2测试。
Anchor草案未通过时留在`drafts/`，不得复制到`frozen/`。历史、QA和Pair生成后，
还必须审核`output_review.yaml`；在`approved: true`前，即使候选文件已经写入
`frozen/`目录，也不得标记为正式Benchmark数据。

`source_review.yaml`只保存布尔结果、Reviewer和日期。具体问题写入
同角色目录下的`review_findings.yaml`。存在状态为`open`的`blocker`或`major`
问题时，来源审核和输出审核均不得通过：

```yaml
item_id: anchor_or_profile_field
severity: blocker | major | minor
rule_id: chronology_total_order
evidence: 简短事实或来源位置
required_change: 可直接执行的修改要求
status: open | fixed | accepted
```

## 附件：32个世界与NPC速查

本附件仅帮助Reviewer定位作品、共享快照和角色，不属于通过条件。正式清单以
`configs/datagen/world_catalog.yaml`为准。

### 中文世界

1. `world_002`《潜伏》——快照：采用现有正典定义；NPC：余则成、王翠平、吴敬中、
   李涯、陆桥山、谢若林。测试重点：双重身份、秘密保护、证据门槛、派系、
   贿赂和权限边界。
2. `world_003`《西游记·三打白骨精》——快照：第三次识破白骨精之前；NPC：孙悟空、
   唐僧、猪八戒、沙僧、白骨精、黑狐精、白龙马。测试重点：身份伪装、
   真假证据、师徒信任、服从与独立判断、战斗权限。
3. `world_004`《三国演义·赤壁前夕》——快照：孙刘联盟形成、赤壁决战尚未开始；
   NPC：曹操、刘备、诸葛亮、关羽、张飞、周瑜、孙权。测试重点：联盟、
   军令、计谋、忠诚、情报真伪和资源调度。
4. `world_005`《红楼梦》——快照：抄检大观园之前，具体回目待人工锁定；NPC：贾宝玉、
   林黛玉、薛宝钗、王熙凤、贾母、晴雯、袭人。测试重点：隐性情绪、礼法、
   关系变化、流言、阶层权限和委婉表达。
5. `world_006`《甄嬛传》——快照：甄嬛初次失宠之前，具体集数待人工锁定；NPC：甄嬛、
   皇帝、华妃、皇后、安陵容、沈眉庄、苏培盛。测试重点：宫廷等级、话语含义、
   联盟、秘密、奖惩权限和表里不一。
6. `world_007`《天龙八部》——快照：六名角色均活跃的统一章节待人工锁定；NPC：萧峰、
   段誉、虚竹、慕容复、王语嫣、鸠摩智。测试重点：身份揭示、族群忠诚、
   武力边界、承诺、情感和复国目标。
7. `world_008`《鹿鼎记》——快照：六名角色均活跃的统一章节待人工锁定；NPC：韦小宝、
   康熙、陈近南、双儿、阿珂、苏荃。测试重点：多重阵营、谎言、效忠承诺、
   求生、交易和亲密关系。
8. `world_009`《武林外传》——快照：主要成员已聚齐且莫小贝仍在同福客栈；NPC：佟湘玉、
   白展堂、郭芙蓉、吕秀才、李大嘴、莫小贝。测试重点：群体日常、债务、劳动、
   冲突调解、喜剧风格和低风险任务。
9. `world_010`《亮剑》——快照：李云龙与赵刚共同任职、秀芹仍在世；NPC：李云龙、赵刚、
   楚云飞、孔捷、丁伟、秀芹。测试重点：军令、战术判断、战友情、组织纪律、
   资源使用和风险权衡。
10. `world_011`《狂飙》——快照：2000年除夕冲突后的早期阶段；NPC：安欣、高启强、
    李响、唐小龙、唐小虎、孟钰。测试重点：权力成长、警务边界、腐化诱因、
    家庭关系和证据判断。
11. `world_012`《家有儿女》——快照：重组家庭稳定生活的早期剧集；NPC：刘梅、夏东海、
    刘星、夏雪、夏雨、姥姥。测试重点：家庭规则、代际沟通、教育、善意谎言和
    低风险承诺。
12. `world_013`《唐人街探案》——快照：第一部电影案件调查中段；NPC：秦风、唐仁、
    思诺、阿香、坤泰、黄兰登。测试重点：推理、证据、警务权限、嫌疑与信任、
    利益动机。
13. `world_014`《九品芝麻官》——快照：包龙星开始重查戚家案件之后；NPC：包龙星、
    常威、方唐镜、戚秦氏、来福、豹头。测试重点：司法、伪证、权力压迫、
    证据翻转、承诺和讽刺表达。
14. `world_015`《笑傲江湖》——快照：六名角色状态兼容的版本与章节待人工锁定；
    NPC：令狐冲、任盈盈、岳不群、林平之、东方不败、仪琳。测试重点：门派规则、
    名誉与真实动机、复仇、爱情和武力边界。
15. `world_016`《还珠格格》——快照：紫薇身份已向核心同伴公开、尚未正式确认；
    NPC：小燕子、紫薇、乾隆、永琪、尔康、皇后。测试重点：身份秘密、宫廷规则、
    亲情、友情、权威和越界行动。
16. `world_017`《原神·璃月/稻妻跨区角色集》——快照：共同时间点待锁定，优先考虑拆分；
    NPC：钟离、胡桃、雷电将军、神里绫华、流浪者、派蒙。测试重点：神/人身份、
    地区规则、契约、职责、游戏任务和物品使用。

### 英文世界

1. `world_001` *Harry Potter and the Prisoner of Azkaban*——快照：尖叫棚屋真相揭示前；
   NPC：Harry Potter、Hermione Granger、Ron Weasley、Remus Lupin、
   Sirius Black、Severus Snape。测试重点：时间快照、秘密、证据、偏见以及
   学生/教师权限。
2. `world_018` *Friends*——快照：第一季固定剧集范围；NPC：Rachel Green、
   Monica Geller、Phoebe Buffay、Joey Tribbiani、Chandler Bing、
   Ross Geller、Gunther。测试重点：友情、浪漫关系、隐瞒、承诺、工作和
   日常社交。
3. `world_019` *The Big Bang Theory*——快照：七名角色均已登场的固定剧集范围；
   NPC：Sheldon Cooper、Leonard Hofstadter、Penny、Howard Wolowitz、
   Raj Koothrappali、Amy Farrah Fowler、Bernadette Rostenkowski。
   测试重点：规则执着、社交暗示、知识权威、友谊和伴侣关系。
4. `world_020` *A Game of Thrones*——快照：第一季Eddard Stark被捕前；NPC：
   Eddard Stark、Catelyn Stark、Arya Stark、Tyrion Lannister、Jon Snow、
   Daenerys Targaryen、Cersei Lannister。测试重点：多阵营忠诚、继承权、
   秘密、政治交易和暴力命令。
5. `world_021` *The Lord of the Rings*——快照：护戒队进入摩瑞亚前；NPC：
   Frodo Baggins、Samwise Gamgee、Gandalf、Aragorn、Legolas、Gimli、
   Boromir。测试重点：任务承诺、诱惑、资源保管、领导权和跨族群信任。
6. `world_022` *Sherlock — A Study in Pink*——快照：首案最终对峙前；NPC：
   Sherlock Holmes、John Watson、Greg Lestrade、Mycroft Holmes、
   Molly Hooper、Mrs Hudson。测试重点：推理证据、警方权限、隐私、能力差异和
   信任形成。
7. `world_023` *The Matrix*——快照：Neo加入Nebuchadnezzar后、Cypher背叛前；
   NPC：Neo、Trinity、Morpheus、Agent Smith、Cypher、The Oracle。
   测试重点：现实知识、信念更新、预言解释、背叛、系统权限和行动。
8. `world_024` *Twilight*——快照：Bella得知Cullen家族秘密后、最终追猎前；
   NPC：Bella Swan、Edward Cullen、Jacob Black、Alice Cullen、
   Carlisle Cullen、Rosalie Hale。测试重点：身份秘密、保护与控制、亲密关系和
   超自然能力边界。
9. `world_025` *The Hunger Games*——快照：本作首场饥饿游戏开始前；NPC：
   Katniss Everdeen、Peeta Mellark、Gale Hawthorne、Haymitch Abernathy、
   Effie Trinket、President Snow。测试重点：公开表演与真实立场、生存、联盟、
   资源、牺牲和权威。
10. `world_026` *My Little Pony — Friendship Is Magic*——快照：主角团形成后的
    早期固定剧集范围；NPC：Twilight Sparkle、Applejack、Pinkie Pie、
    Rainbow Dash、Rarity、Fluttershy。测试重点：群体协作、诚实、忠诚、
    冲突修复和语言风格。
11. `world_027` *Lucifer*——快照：第一季固定剧集范围；NPC：Lucifer Morningstar、
    Chloe Decker、Dan Espinoza、Mazikeen、Amenadiel、Linda Martin。
    测试重点：身份真相、欲望、警务证据、治疗隐私和家庭冲突。
12. `world_028` *Hannibal*——快照：第一季固定剧集范围；NPC：Hannibal Lecter、
    Will Graham、Jack Crawford、Alana Bloom、Abigail Hobbs、Beverly Katz。
    测试重点：欺骗、调查、医疗伦理、秘密、操控和不可靠认知。
13. `world_029` *Rick and Morty*——快照：早期固定剧集范围；NPC：Rick Sanchez、
    Morty Smith、Beth Smith、Jerry Smith、Summer Smith、Birdperson。
    测试重点：科技物品、跨世界规则、家庭关系、风险、操控和荒诞风格。
14. `world_030` *Once Upon a Time*——快照：第一季诅咒解除前；NPC：Emma Swan、
    Henry Mills、Regina Mills、Snow White、Prince Charming、
    Rumplestiltskin。测试重点：双重身份、诅咒知识、亲子关系、交易契约和
    魔法规则。
15. `world_031` *Grey's Anatomy*——快照：第一季固定剧集范围；NPC：Meredith Grey、
    Cristina Yang、Derek Shepherd、Izzie Stevens、George O'Malley、
    Miranda Bailey。测试重点：医疗权限、职业伦理、团队协作、隐私和感情冲突。
16. `world_032` *How I Met Your Mother*——快照：第一季固定剧集范围；NPC：
    Ted Mosby、Robin Scherbatsky、Barney Stinson、Lily Aldrin、
    Marshall Eriksen、Ranjit。测试重点：友情、关系承诺、叙事视角、社交计划和
    隐瞒。
