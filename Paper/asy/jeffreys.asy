settings.outformat="pdf";
size(17.5cm);
import roundedpath;

real box_size = 0.6;
real vert_gap = 0.5;
real horz_gap = 0.4;
real total_len = 3 * box_size + 2 * vert_gap;

real corner_radius = 0.15; // Define the corner radius for rounding

void Draw_Box(pair center, string lbl, pen bg_color=white) {
    pair bl = center - (box_size / 2, box_size / 2);
    pair tr = center + (box_size / 2, box_size / 2);
    path b = roundedpath(box(bl, tr), corner_radius);
    filldraw(b, bg_color, black);
    label(lbl, center);
}

void Draw_Rec(pair center, string lbl, pen bg_color=white) {
    pair bl = center - (box_size / 2, total_len / 2);
    pair tr = center + (box_size / 2, total_len / 2);
    path b = roundedpath(box(bl, tr), corner_radius);
    filldraw(b, bg_color, black);
    label(lbl, center);
}

void Draw_Resample_Arrow(real x) {
    pair nu_top = (x, -box_size/2 - vert_gap);
    pair mu_prime_bottom = (x, -box_size/2);
    draw(nu_top -- mu_prime_bottom, Arrow);
    label("\hspace{-2pt}resample", (nu_top + mu_prime_bottom)/2, RightSide);
}

void Draw_Train_Arrow(real x) {
    pair mu_right = (x + box_size / 2, box_size + vert_gap);
    pair mu_prime_right = (x + box_size / 2, 0);

    real avg_y = (box_size + vert_gap) / 2;
    pair flow_left_mid = (x + horz_gap + box_size / 2, avg_y);

    draw(mu_right -- flow_left_mid, Arrow);
    draw(mu_prime_right -- flow_left_mid, Arrow);

    pair top_mid = (mu_right + flow_left_mid) / 2;
    pair bottom_mid = (mu_prime_right + flow_left_mid) / 2;
    label("train~~~~", (top_mid + bottom_mid) / 2);
}

void Draw_Pushforward(real cx1, real cx2, real cx3) {
    real nu_bottom_y = -box_size - vert_gap - box_size/2;
    real line_bottom_y = nu_bottom_y - vert_gap;

    // Draw the left and middle downward lines
    draw((cx1, nu_bottom_y) -- (cx1, line_bottom_y));
    draw((cx2, nu_bottom_y) -- (cx2, line_bottom_y));

    // Draw the continuous bottom connecting line
    draw((cx1, line_bottom_y) -- (cx3, line_bottom_y));

    // Draw the upward-pointing arrow for the right line
    draw((cx3, line_bottom_y) -- (cx3, nu_bottom_y), Arrow);

    // Add labels "pushforward" and "reweighting" on the segment l23
    pair mid_23 = ((cx2 + cx3) / 2, line_bottom_y);
    label("pushforward", mid_23, N);
    label("reweighting", mid_23, S);
}

void Draw_Dots(real cx) {
    label("$\cdots$", (cx, box_size + vert_gap));
    label("$\cdots$", (cx, 0));
    label("$\cdots$", (cx, -box_size - vert_gap));
}

void Draw_PT(real[] cx) {
    real mu_top_y = box_size + vert_gap + box_size/2;
    real pt_top_y = mu_top_y + vert_gap;

    // Draw the downward arrows to the mu boxes
    for(int i=0; i<cx.length; ++i) {
        draw((cx[i], pt_top_y) -- (cx[i], mu_top_y), Arrow);
    }

    // Draw the continuous top connecting line
    draw((cx[0], pt_top_y) -- (cx[cx.length-1], pt_top_y));

    // Add label above the center
    label("Parallel Tempering", ((cx[0] + cx[cx.length-1]) / 2, pt_top_y), N);
}

// Color fade parameters for customizable box backgrounds (0.0 = white, 1.0 = full color)
real fade_blue = 0.25;
real fade_red = 0.25;
pen custom_lightblue = fade_blue * lightblue + (1 - fade_blue) * white;
pen custom_lightred = fade_red * lightred + (1 - fade_red) * white;

// Source distributions (mu_1, nu_0)
Draw_Box((0, box_size + vert_gap), "$\mu_1$", custom_lightblue);
Draw_Box((0, 0), "$\mu_0'$", custom_lightred);
Draw_Box((0, -box_size - vert_gap), "\shortstack{$\nu_0$ \\[4pt] (base)}", custom_lightred);

Draw_Resample_Arrow(0);
Draw_Train_Arrow(0);

real pf_x1 = box_size / 6;
real pf_x2 = horz_gap + box_size;
real pf_x3 = 2 * (box_size + horz_gap) - box_size / 6;
Draw_Pushforward(pf_x1, pf_x2, pf_x3);

// Flow F_1
Draw_Rec((horz_gap + box_size, 0), "\shortstack{flow \\[4pt] $F_1$}");

// Stage 1 distributions (mu_2, nu_1)
real x1 = 2 * (box_size + horz_gap);
Draw_Box((x1, box_size + vert_gap), "$\mu_2$", custom_lightblue);
Draw_Box((x1, 0), "$\mu_1'$", custom_lightred);
Draw_Box((x1, -box_size - vert_gap), "$\nu_1$", custom_lightred);

Draw_Resample_Arrow(x1);
Draw_Train_Arrow(x1);

// Flow F_2
Draw_Rec((x1 + horz_gap + box_size, 0), "\shortstack{flow \\[4pt] $F_2$}");

// Dots
real dots_x = x1 + 1.5 * box_size + 2 * horz_gap;
Draw_Dots(dots_x);

// Stage M-1 distributions (now labeled mu_M)
real x_Mm1 = dots_x + horz_gap + box_size / 2;
Draw_Box((x_Mm1, box_size + vert_gap), "$\mu_M$", custom_lightblue);
Draw_Box((x_Mm1, 0), "$\mu_{M-1}'$", custom_lightred);
Draw_Box((x_Mm1, -box_size - vert_gap), "$\nu_{M-1}$", custom_lightred);

Draw_Resample_Arrow(x_Mm1);
Draw_Train_Arrow(x_Mm1);

// Flow F_M
Draw_Rec((x_Mm1 + horz_gap + box_size, 0), "\shortstack{flow \\[4pt] $F_M$}");

// Stage M distributions (Only nu_M remains, mu_M is removed)
real xM = x_Mm1 + 2 * (box_size + horz_gap);
Draw_Box((xM, -box_size - vert_gap), "\shortstack{$\nu_M$ \\[4pt] (target)}", custom_lightred);

// Pushforward arrow from nu_1 via F_2 to nu_{M-1} (left 1/3 pt)
real pf2_x1 = x1 + box_size / 6;
real pf2_x2 = x1 + horz_gap + box_size;
real pf2_x3 = x_Mm1 - box_size / 6;
Draw_Pushforward(pf2_x1, pf2_x2, pf2_x3);

// Pushforward arrow from nu_{M-1} via F_M to nu_M
real pfM_x1 = x_Mm1 + box_size / 6;
real pfM_x2 = x_Mm1 + horz_gap + box_size;
real pfM_x3 = xM - box_size / 6;
Draw_Pushforward(pfM_x1, pfM_x2, pfM_x3);

// Draw Parallel Tempering line connecting 0, x1, and x_{M-1} only
Draw_PT(new real[]{0, x1, x_Mm1});