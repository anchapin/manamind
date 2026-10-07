package manamind.forge;

import forge.LobbyPlayer;
import forge.ai.*;
import forge.deck.Deck;
import forge.deck.io.DeckSerializer;
import forge.game.*;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import forge.model.FModel;
import forge.player.GamePlayerUtil;
import java.io.*;
import java.util.*;

/**
 * One seat is driven by an external process over line-delimited JSON.
 * Protocol lines on stdout start with "@@MM " (Forge also logs to stdout);
 * replies on stdin are one line each:
 *   priority: an option index, where index == options.length means pass
 *   attack:   space-separated indices of creatures to attack with (may be empty)
 *   block:    space-separated "blocker:attacker" index pairs (may be empty)
 */
public class PipeBench {
    static final String TAG = "@@MM ";
    static PrintStream proto;
    static BufferedReader in;
    static long decisions = 0, fallbacks = 0, errors = 0;

    static String q(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if (c == '"' || c == '\\') b.append('\\').append(c);
            else if (c < 0x20) b.append(' ');
            else b.append(c);
        }
        return b.append('"').toString();
    }

    static String names(List<Card> cs) {
        StringJoiner j = new StringJoiner(",", "[", "]");
        for (Card c : cs) j.add(q(c.getName()));
        return j.toString();
    }

    static String card(Card c) {
        return "{\"name\":" + q(c.getName())
                + ",\"type\":" + q(String.valueOf(c.getType()))
                + ",\"cost\":" + q(String.valueOf(c.getManaCost()))
                + ",\"cmc\":" + c.getCMC()
                + ",\"creature\":" + c.isCreature()
                + ",\"land\":" + c.isLand()
                + ",\"power\":" + (c.isCreature() ? c.getNetPower() : 0)
                + ",\"toughness\":" + (c.isCreature() ? c.getNetToughness() : 0)
                + ",\"tapped\":" + c.isTapped()
                + ",\"sick\":" + c.isSick() + "}";
    }

    static String cards(List<Card> cs) {
        StringJoiner j = new StringJoiner(",", "[", "]");
        for (Card c : cs) j.add(card(c));
        return j.toString();
    }

    static String view(Game g, Player me) {
        Player opp = me.getSingleOpponent();
        return "\"turn\":" + g.getPhaseHandler().getTurn()
                + ",\"phase\":" + q(String.valueOf(g.getPhaseHandler().getPhase()))
                + ",\"active\":" + (g.getPhaseHandler().getPlayerTurn() == me)
                + ",\"life\":[" + me.getLife() + "," + opp.getLife() + "]"
                + ",\"hand\":" + cards(new ArrayList<>(me.getCardsIn(ZoneType.Hand)))
                + ",\"opp_hand_size\":" + opp.getCardsIn(ZoneType.Hand).size()
                + ",\"battlefield\":" + cards(new ArrayList<>(me.getCardsIn(ZoneType.Battlefield)))
                + ",\"opp_battlefield\":" + cards(new ArrayList<>(opp.getCardsIn(ZoneType.Battlefield)))
                + ",\"graveyard\":" + names(new ArrayList<>(me.getCardsIn(ZoneType.Graveyard)))
                + ",\"opp_graveyard\":" + names(new ArrayList<>(opp.getCardsIn(ZoneType.Graveyard)))
                + ",\"library\":[" + me.getCardsIn(ZoneType.Library).size() + ","
                + opp.getCardsIn(ZoneType.Library).size() + "]";
    }

    static String ask(String json) {
        proto.println(TAG + json);
        proto.flush();
        try {
            String line = in.readLine();
            if (line == null) throw new IllegalStateException("driver closed stdin");
            return line.trim();
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }

    static class PipeController extends PlayerControllerAi {
        PipeController(Game g, Player p, LobbyPlayer lp) { super(g, p, lp); }

        @Override
        public List<SpellAbility> chooseSpellAbilityToPlay() {
            decisions++;
            Player me = getPlayer();
            List<SpellAbility> legal = new ArrayList<>();
            try {
                List<SpellAbility> all = ComputerUtilAbility.getSpellAbilities(
                        ComputerUtilAbility.getAvailableCards(getGame(), me), me);
                all = ComputerUtilAbility.getOriginalAndAltCostAbilities(all, me);
                for (SpellAbility sa : all) {
                    try {
                        if (sa.canPlay() && getAi().canPlaySa(sa) == AiPlayDecision.WillPlay) legal.add(sa);
                    } catch (RuntimeException e) { errors++; }
                }
            } catch (RuntimeException e) { errors++; }
            StringJoiner opts = new StringJoiner(",", "[", "]");
            for (SpellAbility sa : legal)
                opts.add("{\"text\":" + q(sa.getHostCard().getName() + " | " + sa.toString())
                        + ",\"land\":" + sa.isLandAbility() + ",\"spell\":" + sa.isSpell()
                        + ",\"card\":" + card(sa.getHostCard()) + "}");
            String r = ask("{\"t\":\"priority\"," + view(getGame(), me) + ",\"options\":" + opts + "}");
            int k;
            try { k = Integer.parseInt(r); } catch (NumberFormatException e) { k = legal.size(); }
            if (k < 0 || k >= legal.size()) return null;
            return Collections.singletonList(legal.get(k));
        }

        @Override
        public void declareAttackers(Player attacker, Combat combat) {
            decisions++;
            if (combat.getDefenders().isEmpty()) return;
            var def = combat.getDefenders().get(0);
            List<Card> can = new ArrayList<>();
            for (Card c : attacker.getCreaturesInPlay()) if (CombatUtil.canAttack(c, def)) can.add(c);
            if (can.isEmpty()) return;
            String r = ask("{\"t\":\"attack\"," + view(getGame(), attacker) + ",\"options\":" + cards(can) + "}");
            for (String tok : r.split("\\s+")) {
                if (tok.isEmpty()) continue;
                try {
                    int i = Integer.parseInt(tok);
                    if (i >= 0 && i < can.size()) combat.addAttacker(can.get(i), def);
                } catch (NumberFormatException e) { errors++; }
            }
            if (!CombatUtil.validateAttackers(combat)) {
                fallbacks++;
                for (Card c : new ArrayList<>(combat.getAttackers())) combat.removeFromCombat(c);
                super.declareAttackers(attacker, combat);
            }
        }

        @Override
        public void declareBlockers(Player defender, Combat combat) {
            decisions++;
            List<Card> attackers = new ArrayList<>(combat.getAttackers());
            List<Card> blockers = new ArrayList<>(defender.getCreaturesInPlay());
            if (attackers.isEmpty() || blockers.isEmpty()) return;
            String r = ask("{\"t\":\"block\"," + view(getGame(), defender) + ",\"attackers\":" + cards(attackers)
                    + ",\"blockers\":" + cards(blockers) + "}");
            List<Card> added = new ArrayList<>();
            for (String tok : r.split("\\s+")) {
                String[] p = tok.split(":");
                if (p.length != 2) continue;
                try {
                    int b = Integer.parseInt(p[0]), a = Integer.parseInt(p[1]);
                    if (b >= 0 && b < blockers.size() && a >= 0 && a < attackers.size()
                            && !added.contains(blockers.get(b))
                            && CombatUtil.canBlock(attackers.get(a), blockers.get(b), combat)) {
                        combat.addBlocker(attackers.get(a), blockers.get(b));
                        added.add(blockers.get(b));
                    }
                } catch (NumberFormatException e) { errors++; }
            }
            if (CombatUtil.validateBlocks(combat, defender) != null) {
                fallbacks++;
                for (Card b : added) combat.removeFromCombat(b);
                super.declareBlockers(defender, combat);
            }
        }
    }

    static class PipeLobby extends LobbyPlayerAi {
        PipeLobby(String name) { super(name, EnumSet.noneOf(AIOption.class)); }

        @Override
        public Player createIngamePlayer(Game game, int id) {
            Player p = new Player(getName(), game, id);
            p.setFirstController(new PipeController(game, p, this));
            return p;
        }
    }

    public static void main(String[] args) throws Exception {
        proto = new PrintStream(new FileOutputStream(FileDescriptor.out), false, "UTF-8");
        in = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        int games = Integer.parseInt(args[0]);
        Deck a = DeckSerializer.fromFile(new File(args[1]));
        Deck b = DeckSerializer.fromFile(new File(args[2]));
        System.setProperty("java.awt.headless", "true");
        forge.gui.GuiBase.setInterface(new forge.GuiDesktop());
        FModel.initialize(null, null);
        proto.println(TAG + "{\"t\":\"ready\"}");
        proto.flush();
        for (int i = 0; i < games; i++) {
            boolean pipeFirst = (i % 2 == 0);
            Deck pd = (i / 2) % 2 == 0 ? a : b, ad = pd == a ? b : a;
            PipeLobby pl = new PipeLobby("Manamind");
            LobbyPlayer al = GamePlayerUtil.createAiPlayer("ForgeAI", 0);
            RegisteredPlayer r1 = new RegisteredPlayer(pd).setPlayer(pl);
            RegisteredPlayer r2 = new RegisteredPlayer(ad).setPlayer(al);
            List<RegisteredPlayer> ps = pipeFirst ? List.of(r1, r2) : List.of(r2, r1);
            GameRules rules = new GameRules(GameType.Constructed);
            rules.setGamesPerMatch(1);
            Match m = new Match(rules, new ArrayList<>(ps), "pipe" + i);
            long t0 = System.currentTimeMillis();
            String result;
            int turns = -1;
            try {
                Game g = m.createGame();
                g.setNoGUIUser();
                m.startGame(g);
                GameOutcome o = g.getOutcome();
                turns = o == null ? -1 : o.getLastTurnNumber();
                result = o == null || o.isDraw() ? "draw" : (o.getWinningLobbyPlayer() == pl ? "win" : "loss");
            } catch (Throwable e) {
                result = "crash";
                e.printStackTrace(System.err);
            }
            proto.println(TAG + "{\"t\":\"game_over\",\"game\":" + i + ",\"result\":" + q(result)
                    + ",\"turns\":" + turns + ",\"ms\":" + (System.currentTimeMillis() - t0)
                    + ",\"deck\":" + q(pd.getName()) + ",\"on_play\":" + pipeFirst + "}");
            proto.flush();
        }
        proto.println(TAG + "{\"t\":\"done\",\"decisions\":" + decisions + ",\"fallbacks\":" + fallbacks
                + ",\"errors\":" + errors + "}");
        proto.flush();
        System.exit(0);
    }
}
