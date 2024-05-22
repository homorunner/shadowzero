package gating

import (
	"context"
	"ddt/pkg/repo"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/jackc/pgx/v5"
	"github.com/pelletier/go-toml/v2"
)

type Repo struct {
	db *pgx.Conn
}

func New(args repo.RepoArgs) *Repo {
	return &Repo{
		db: args.DB,
	}
}

func (r *Repo) Handle(ctx context.Context, action string, args []string) {
	switch action {
	case "addresult":
		r.addresult(ctx, args)
	case "showelo":
		r.showelo(ctx)
	}
}

type GatingResult struct {
	ModelResults []ModelResult `toml:"model"`
}

type ModelResult struct {
	ModelPath       string  `toml:"path"`
	FirstplayCount  int     `toml:"firstplay_count"`
	FirstplayScore  float64 `toml:"firstplay_score"`
	SecondplayCount int     `toml:"secondplay_count"`
	SecondplayScore float64 `toml:"secondplay_score"`
}

func (r *Repo) addresult(ctx context.Context, args []string) {
	if len(args) < 1 {
		log.Fatal("not enough arguments for addresult")
	}

	resultFile := args[0]
	if _, err := os.Stat(resultFile); os.IsNotExist(err) {
		log.Fatal(err)
	}

	content, err := os.ReadFile(resultFile)
	if err != nil {
		log.Fatal(err)
	}
	var gatingResult GatingResult
	err = toml.Unmarshal(content, &gatingResult)
	if err != nil {
		log.Fatal(err)
	}

	if gatingResult.ModelResults[0].FirstplayCount != gatingResult.ModelResults[1].SecondplayCount {
		log.Fatal("firstplay count != secondplay count, gating result invalid")
	}

	for i := range 2 {
		p1 := filepath.Base(gatingResult.ModelResults[i].ModelPath)
		p2 := filepath.Base(gatingResult.ModelResults[1-i].ModelPath)
		score := gatingResult.ModelResults[i].FirstplayScore
		count := gatingResult.ModelResults[i].FirstplayCount
		_, err = r.db.Exec(ctx, `INSERT INTO gatingresult (train_id, p1, p2, score, count) VALUES ($1, $2, $3, $4, $5) ON CONFLICT (train_id, p1, p2) DO UPDATE SET score=gatingresult.score+$4, count=gatingresult.count+$5`, "", p1, p2, score, count)
		if err != nil {
			log.Fatal(err)
		}

		log.Print("add result to db ok.")
	}
}

func (r *Repo) showelo(ctx context.Context) {
	rows, err := r.db.Query(ctx, "SELECT p1, p2, score, count FROM gatingresult where train_id = $1", "")
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	eloCommandline := []string{
		"python/elo.py",
	}

	for rows.Next() {
		var p1, p2 string
		var score float64
		var count int
		err = rows.Scan(&p1, &p2, &score, &count)
		if err != nil {
			log.Fatal(err)
		}
		eloCommandline = append(eloCommandline, p1, p2, fmt.Sprint(score), fmt.Sprint(count))
	}

	cmd := exec.Command("python3", eloCommandline...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		log.Fatal(err)
	}
}
