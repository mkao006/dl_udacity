# Create the visualisation of the Heuristic Analysis.
library(ggplot2)

adjustment = 1.25

# Function for multi-ggplot
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

legalMoveCount <- function(x, y){
    dx = c(1, 2, 2, 1, -1, -2, -2, -1)
    dy = c(2, 1, -1, -2, -2, -1, 1, 2)
    legalMoves = (x + dx >= 1 & x + dx <= 7 & y + dy >= 1 & y + dy <= 7)
    scaler = length(dx) * adjustment
    sum(legalMoves)/ scaler
}



# Define the grid
boardColIndex = rep(1:7, 7)
boardRowIndex = rep(1:7, each=7)


# Visualise the uniform board
improvedBoard = data.frame(boardColIndex = boardColIndex,
                        boardRowIndex = boardRowIndex,
                        score = 1)
improvedBoardPlot =
    ggplot(data = improvedBoard,
       aes(x = boardColIndex, y = boardRowIndex, fill = score)) +
    geom_tile() +
    geom_text(aes(label = paste0("(", boardRowIndex - 1, ", ", boardColIndex - 1, ") - ", score)),
              col = "white") +
    scale_y_reverse() +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank()) +
    labs(title = "a) Improved Player")


# Create the legal move board
legalBoard = data.frame(boardColIndex = boardColIndex,
                        boardRowIndex = boardRowIndex,
                        score = mapply(legalMoveCount, x = boardRowIndex, y = boardColIndex))
legalBoardPlot =
    ggplot(data = legalBoard,
       aes(x = boardColIndex, y = boardRowIndex, fill = score)) +
    geom_tile() +
    geom_text(aes(label = paste0("(", boardRowIndex - 1, ", ", boardColIndex - 1, ") - ", score)),
              col = "white") +
    scale_y_reverse() +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank()) +
    labs(title = "b) Legal Moves")


# Create the Manhattan board
manhattanBoard = data.frame(boardColIndex = boardColIndex,
                            boardRowIndex = boardRowIndex,
                            score = round(1 / (abs(boardColIndex - 4) +
                                               abs(boardRowIndex - 4) + adjustment), 2))
manhattenBoardPlot =
    ggplot(data = manhattanBoard,
       aes(x = boardColIndex, y = boardRowIndex, fill = score)) +
    geom_tile() +
    geom_text(aes(label = paste0("(", boardRowIndex - 1, ", ", boardColIndex - 1, ") - ", score)),
              col = "white") +
    scale_y_reverse() +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank()) +
    labs(title = "c) Manhattan")

# Create the Euclidean board
euclideanBoard = data.frame(boardColIndex = boardColIndex,
                            boardRowIndex = boardRowIndex,
                            score = round(1/ ((boardColIndex - 4)^2 +
                                              (boardRowIndex - 4)^2 + adjustment), 2))
euclideanBoardPlot =
    ggplot(data = euclideanBoard,
       aes(x = boardColIndex, y = boardRowIndex, fill = score)) +
    geom_tile() +
    geom_text(aes(label = paste0("(", boardRowIndex - 1, ", ", boardColIndex - 1, ") - ", score)),
              col = "white") +
    scale_y_reverse() +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank()) +
    labs(title = "d) Euclidean")


# Save the graph
png(filename = "board_position_score.png", width = 1200, height = 1200)
multiplot(improvedBoardPlot, legalBoardPlot, manhattenBoardPlot, euclideanBoardPlot, cols = 2)
graphics.off()
