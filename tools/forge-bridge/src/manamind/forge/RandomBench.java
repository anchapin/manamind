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
import forge.model.FModel;
import forge.player.GamePlayerUtil;
import java.io.File;
import java.util.*;

/** Random-policy seat vs Forge AI: measures decisions/s and exceptions. */
public class RandomBench {
    static final Random RNG = new Random(0);
    static long decisions = 0, fallbacks = 0, errors = 0, ourNs = 0;

    static class RandomController extends PlayerControllerAi {
        RandomController(Game g, Player p, LobbyPlayer lp) { super(g, p, lp); }

        @Override
        public List<SpellAbility> chooseSpellAbilityToPlay() {
            decisions++;
            long t0 = System.nanoTime();
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
            ourNs += System.nanoTime() - t0;
            int k = RNG.nextInt(legal.size() + 1);
            if (k == legal.size()) return null; // pass priority
            return Collections.singletonList(legal.get(k));
        }

        @Override
        public void declareAttackers(Player attacker, Combat combat) {
            decisions++;
            if (combat.getDefenders().isEmpty()) return;
            var def = combat.getDefenders().get(0);
            for (Card c : attacker.getCreaturesInPlay()) {
                if (RNG.nextBoolean() && CombatUtil.canAttack(c, def)) combat.addAttacker(c, def);
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
            if (attackers.isEmpty()) return;
            List<Card> added = new ArrayList<>();
            for (Card b : defender.getCreaturesInPlay()) {
                int k = RNG.nextInt(attackers.size() + 1);
                if (k < attackers.size() && CombatUtil.canBlock(attackers.get(k), b, combat)) {
                    combat.addBlocker(attackers.get(k), b);
                    added.add(b);
                }
            }
            if (CombatUtil.validateBlocks(combat, defender) != null) {
                fallbacks++;
                for (Card b : added) combat.removeFromCombat(b);
                super.declareBlockers(defender, combat);
            }
        }
    }

    static class RandomLobby extends LobbyPlayerAi {
        RandomLobby(String name) { super(name, EnumSet.noneOf(AIOption.class)); }

        @Override
        public Player createIngamePlayer(Game game, int id) {
            Player p = new Player(getName(), game, id);
            p.setFirstController(new RandomController(game, p, this));
            return p;
        }
    }

    public static void main(String[] args) throws Exception {
        int games = Integer.parseInt(args[0]);
        Deck a = DeckSerializer.fromFile(new File(args[1]));
        Deck b = DeckSerializer.fromFile(new File(args[2]));
        System.setProperty("java.awt.headless", "true");
        forge.gui.GuiBase.setInterface(new forge.GuiDesktop());
        FModel.initialize(null, null);
        int randWins = 0, aiWins = 0, draws = 0, crashes = 0;
        long totalMs = 0;
        for (int i = 0; i < games; i++) {
            boolean randFirst = (i % 2 == 0);
            Deck rd = (i / 2) % 2 == 0 ? a : b, ad = rd == a ? b : a;
            RandomLobby rl = new RandomLobby("Random");
            LobbyPlayer al = GamePlayerUtil.createAiPlayer("ForgeAI", 0);
            List<RegisteredPlayer> ps = new ArrayList<>();
            RegisteredPlayer r1 = new RegisteredPlayer(rd).setPlayer(rl);
            RegisteredPlayer r2 = new RegisteredPlayer(ad).setPlayer(al);
            if (randFirst) { ps.add(r1); ps.add(r2); } else { ps.add(r2); ps.add(r1); }
            GameRules rules = new GameRules(GameType.Constructed);
            rules.setGamesPerMatch(1);
            Match m = new Match(rules, ps, "bench" + i);
            long t0 = System.currentTimeMillis();
            long d0 = decisions;
            try {
                Game g = m.createGame();
                g.setNoGUIUser();
                m.startGame(g);
                GameOutcome o = g.getOutcome();
                if (o == null || o.isDraw()) draws++;
                else if (o.getWinningLobbyPlayer() == rl) randWins++;
                else aiWins++;
                System.out.printf("game %d turns=%d ms=%d decisions=%d winner=%s%n", i,
                        o == null ? -1 : o.getLastTurnNumber(), System.currentTimeMillis() - t0,
                        decisions - d0, o == null ? "none" : (o.isDraw() ? "draw" : o.getWinningLobbyPlayer().getName()));
            } catch (Throwable e) {
                crashes++;
                System.out.println("game " + i + " CRASH " + e);
                e.printStackTrace(System.out);
            }
            totalMs += System.currentTimeMillis() - t0;
        }
        System.out.printf("SUMMARY games=%d random=%d ai=%d draws=%d crashes=%d decisions=%d fallbacks=%d errors=%d secs=%.1f decisions/s=%.0f ourSecs=%.1f%n",
                games, randWins, aiWins, draws, crashes, decisions, fallbacks, errors, totalMs / 1000.0, decisions * 1000.0 / totalMs, ourNs / 1e9);
        System.exit(0);
    }
}
